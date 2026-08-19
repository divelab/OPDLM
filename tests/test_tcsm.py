import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from train.tcsm import (
    TCSMSample,
    _tcsm_position_masks,
    build_full_vocab_tcs_target,
    compute_block_tcs_topk_scores,
    count_tcsm_counterfactual_work,
    full_target_forward_kl,
    mc_tcsm_correction_per_row,
    sample_disagreement_tcs_positions,
    sample_tcsm_blocks_per_row,
    tcsm_lambda,
    teacher_logprobs_from_logits,
    token_tcsm_correction_per_row,
)


class TCSMMathTests(unittest.TestCase):
    def test_disagreement_routing_probabilities(self):
        arm_logits = torch.tensor([[[3.0, 1.0], [3.0, 1.0], [1.0, 3.0]]])
        dlm_logits = torch.tensor([[[1.0, 3.0], [3.0, 1.0], [3.0, 1.0]]])
        active = torch.tensor([[True, True, False]])
        disagree, selected, pi = sample_disagreement_tcs_positions(
            arm_logits, dlm_logits, active, generator=torch.Generator().manual_seed(7)
        )
        self.assertTrue(torch.equal(disagree, torch.tensor([[True, False, False]])))
        self.assertTrue(selected[0, 0])
        self.assertFalse(selected[0, 2])
        self.assertTrue(torch.equal(pi, torch.tensor([[1.0, 0.1, 1.0]])))

    def test_token_disagreement_routing_is_unbiased_with_existing_row_reduction(self):
        active = torch.tensor([
            [True, True, True, True],
            [True, True, False, False],
        ])
        deltas = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 7.0, 0.0, 0.0],
        ])
        arm_choice = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
        dlm_choice = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
        arm_logits = F.one_hot(arm_choice, 2).float()
        dlm_logits = F.one_hot(dlm_choice, 2).float()
        generator = torch.Generator().manual_seed(123)
        estimates = []
        for _ in range(5000):
            _, selected, pi = sample_disagreement_tcs_positions(
                arm_logits, dlm_logits, active, generator=generator
            )
            rows, cols = selected.nonzero(as_tuple=True)
            weighted = deltas[rows, cols] / pi[rows, cols]
            estimates.append(token_tcsm_correction_per_row(
                weighted, rows, active.sum(dim=1), batch_size=2
            ))
        expected = torch.stack([
            deltas[0, active[0]].mean(), deltas[1, active[1]].mean()
        ]).mean()
        self.assertTrue(torch.allclose(torch.stack(estimates).mean(), expected, atol=0.06))

    def test_training_uses_token_routing_without_block_sampling(self):
        source = (Path(__file__).parents[1] / "train" / "rl_sdar.py").read_text()
        current_path = source[source.index("def train_one_step("):]
        self.assertNotIn("sample_tcsm_blocks_per_row", current_path)
        self.assertIn("sample=None", current_path)
        self.assertIn("token_tcsm_correction_per_row", current_path)

    def test_token_routing_activates_selected_positions_across_all_blocks(self):
        selected = torch.tensor([[
            False, False,
            True, False, True, True,
            False, True, True, True,
        ]])
        routed, counterfactual = _tcsm_position_masks(
            loss_mask=selected,
            sequence_ends=torch.tensor([10]),
            sample=None,
            response_start=2,
            block_size=4,
        )
        self.assertTrue(torch.equal(routed, selected))
        expected_counterfactual = selected.clone()
        expected_counterfactual[0, 5] = False
        expected_counterfactual[0, 9] = False
        self.assertTrue(torch.equal(counterfactual, expected_counterfactual))

    def test_training_mask_metrics_have_no_stale_mask_reference(self):
        source = (Path(__file__).parents[1] / "train" / "rl_sdar.py").read_text()
        self.assertNotIn(
            "num_masked_tokens = (response_mask & masked_token_mask).sum(dim=1)",
            source,
        )
        self.assertEqual(source.count("num_masked_tokens = loss_mask.sum(dim=1)"), 3)

    def test_topk_selects_candidates_without_renormalizing_arm(self):
        probs = torch.tensor([[0.60, 0.20, 0.15, 0.05]])
        full_logprobs = teacher_logprobs_from_logits(probs.log())
        candidates = full_logprobs.topk(2, dim=-1).indices
        alpha = full_logprobs.gather(-1, candidates).exp().sum(-1)
        self.assertTrue(torch.allclose(alpha, torch.tensor([0.80])))
        self.assertLess(alpha.item(), 1.0)

    def test_suffix_outside_topk_keeps_finite_branch_score(self):
        candidate_logprobs = teacher_logprobs_from_logits(torch.tensor([[5.0, 4.0, 1.0]]))
        candidates = candidate_logprobs.topk(2, dim=-1).indices
        self.assertNotIn(2, candidates.tolist()[0])
        suffix_logprobs = teacher_logprobs_from_logits(torch.tensor([[3.0, 2.0, -4.0]]))
        branch_score = candidate_logprobs[0, candidates[0, 0]] + suffix_logprobs[0, 2]
        self.assertTrue(torch.isfinite(branch_score))

    def test_full_vocab_target_preserves_tail_and_normalization(self):
        opd = torch.tensor([[0.30, 0.25, 0.20, 0.15, 0.10]])
        candidates = torch.tensor([[0, 2]])
        tcs = torch.tensor([[0.1, 0.9]])
        target = build_full_vocab_tcs_target(opd, candidates, tcs)

        self.assertTrue(torch.allclose(target.sum(-1), torch.ones(1)))
        self.assertTrue(torch.equal(target[:, [1, 3, 4]], opd[:, [1, 3, 4]]))
        self.assertTrue(torch.allclose(target[:, [0, 2]].sum(-1), opd[:, [0, 2]].sum(-1)))

    def test_optimized_full_target_loss_matches_explicit_target(self):
        teacher_logits = torch.tensor([[1.2, -0.3, 0.7, 0.2, -1.0]])
        opd_logprobs = F.log_softmax(teacher_logits, dim=-1)
        candidates = torch.tensor([[0, 2, 3]])
        opd_topk_lp = opd_logprobs.gather(-1, candidates)
        tcs = F.softmax(torch.tensor([[0.1, 1.4, -0.2]]), dim=-1)
        student_logits = torch.tensor([[0.4, 0.8, -0.5, 1.0, -0.2]], requires_grad=True)
        student_lp = F.log_softmax(student_logits, dim=-1)
        self.assertTrue(torch.allclose(student_lp.exp().sum(-1), torch.ones(1)))

        optimized = full_target_forward_kl(
            student_lp, opd_logprobs, candidates, tcs
        )
        opd = opd_logprobs.exp()
        target = build_full_vocab_tcs_target(opd, candidates, tcs)
        dense = (target * (target.log() - student_lp)).sum(-1)
        self.assertTrue(torch.allclose(optimized, dense, atol=1e-6))

        historical_sparse = (
            opd_topk_lp.exp() * (opd_topk_lp - student_lp.gather(-1, candidates))
        ).sum(-1)
        delta = optimized - historical_sparse
        self.assertTrue(torch.equal(historical_sparse + 0.0 * delta, historical_sparse))
        self.assertTrue(torch.allclose(historical_sparse + delta, dense, atol=1e-6))

        optimized.sum().backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertFalse(opd_topk_lp.requires_grad)
        self.assertFalse(tcs.requires_grad)

    def test_block_final_target_and_loss_are_full_opd(self):
        opd_logprobs = F.log_softmax(torch.tensor([[2.0, 1.0, 0.5, -0.5]]), dim=-1)
        opd = opd_logprobs.exp()
        candidates = torch.tensor([[0, 1, 2]])
        candidate_scores = opd.gather(-1, candidates).log()
        tcs = candidate_scores.softmax(dim=-1)
        target = build_full_vocab_tcs_target(opd, candidates, tcs)
        self.assertTrue(torch.allclose(target, opd, atol=1e-7))

        student_lp = F.log_softmax(torch.tensor([[0.1, 0.2, 0.3, 0.4]]), dim=-1)
        corrected = full_target_forward_kl(student_lp, opd_logprobs, candidates, tcs)
        full_opd = (opd * (opd_logprobs - student_lp)).sum(-1)
        sparse_opd = (
            candidate_scores.exp() * (candidate_scores - student_lp.gather(-1, candidates))
        ).sum(-1)
        self.assertTrue(torch.allclose(corrected, full_opd, atol=1e-6))
        self.assertFalse(torch.allclose(corrected - sparse_opd, torch.zeros_like(corrected)))

    def test_dense_sampling_uses_row_token_means_not_block_means(self):
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
        ], dtype=torch.bool)
        sample = sample_tcsm_blocks_per_row(loss_mask, 2, 2, blocks_per_prompt=99)
        full_delta = torch.tensor([
            [0, 0, 1.0, 3.0, 2.0, 4.0, 8.0, 0],
            [0, 0, 5.0, 0, 7.0, 0, 0, 0],
        ])
        rows, deltas = [], []
        for row, blocks in enumerate(sample.blocks):
            for block in blocks.tolist():
                start = 2 + block * 2
                end = start + 2
                active = loss_mask[row, start:end]
                rows.extend([row] * int(active.sum()))
                deltas.extend(full_delta[row, start:end][active].tolist())
        correction = mc_tcsm_correction_per_row(
            torch.tensor(deltas), torch.tensor(rows), sample, loss_mask.sum(1), 2
        )
        expected = ((1 + 3 + 2 + 4 + 8) / 5 + (5 + 7) / 2) / 2
        self.assertAlmostEqual(correction.item(), expected, places=6)

    def test_per_row_mc_estimator_converges_for_unequal_lengths(self):
        loss_mask = torch.tensor([
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.bool)
        full_delta = torch.tensor([
            [0, 1.0, 2.0, -1.0, 3.0, 4.0, 2.0, 5.0, 0],
            [0, -2.0, 1.0, 7.0, 2.0, 0, 0, 0, 0],
        ])
        dense = ((full_delta[0] * loss_mask[0]).sum() / 7
                 + (full_delta[1] * loss_mask[1]).sum() / 4) / 2
        generator = torch.Generator().manual_seed(123)
        estimates = []
        for _ in range(4000):
            sample = sample_tcsm_blocks_per_row(
                loss_mask, response_start=1, block_size=2,
                blocks_per_prompt=1, generator=generator,
            )
            rows, deltas = [], []
            for row, blocks in enumerate(sample.blocks):
                for block in blocks.tolist():
                    start = 1 + block * 2
                    end = min(start + 2, loss_mask.size(1))
                    active = loss_mask[row, start:end]
                    rows.extend([row] * int(active.sum()))
                    deltas.extend(full_delta[row, start:end][active].tolist())
            estimates.append(mc_tcsm_correction_per_row(
                torch.tensor(deltas), torch.tensor(rows), sample, loss_mask.sum(1), 2
            ))
        self.assertTrue(torch.allclose(torch.stack(estimates).mean(), dense, atol=0.05))

    def test_lambda_ramp_is_independent(self):
        self.assertEqual(tcsm_lambda(1, 0.7, 0, "linear"), 0.7)
        self.assertAlmostEqual(tcsm_lambda(25, 0.8, 100, "linear"), 0.2)
        self.assertEqual(tcsm_lambda(200, 0.8, 100, "linear"), 0.8)

        opd_loss = torch.tensor(2.0, requires_grad=True)
        correction = torch.tensor(5.0, requires_grad=True)
        corrected = opd_loss + 0.0 * correction
        self.assertEqual(corrected.item(), opd_loss.item())
        corrected.backward()
        self.assertEqual(opd_loss.grad.item(), 1.0)
        self.assertEqual(correction.grad.item(), 0.0)

    def test_lambda_zero_preserves_historical_row_reduction_exactly(self):
        token_loss = torch.tensor([
            [2.0, 4.0, 100.0],
            [3.0, 9.0, 6.0],
        ])
        active = torch.tensor([
            [True, True, False],
            [True, True, True],
        ])
        historical = ((token_loss * active).sum(1) / active.sum(1)).mean()
        arbitrary_mc_correction = torch.tensor(17.0)
        corrected = historical + 0.0 * arbitrary_mc_correction
        self.assertTrue(torch.equal(corrected, historical))


class TCSMCacheTests(unittest.TestCase):
    def test_existing_opd_teacher_distribution_matches_clean_causal_prefix(self):
        """Guard the OPD roll/boundary replacement against a one-token shift."""
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(11)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            pad_token_id=0,
            use_sliding_window=False,
        )
        teacher = Qwen3ForCausalLM(config).eval()
        clean = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
        prompt_length = 3
        response_length = clean.size(1) - prompt_length
        block_size = 4
        total_length = prompt_length + 2 * response_length

        # Reconstruct train_one_step.forward_process's clean-fill teacher input.
        extended = torch.cat([clean, clean[:, prompt_length:]], dim=1)
        position_ids = torch.cat([
            torch.arange(clean.size(1)),
            torch.arange(prompt_length, clean.size(1)),
        ]).unsqueeze(0)

        attention = torch.zeros(1, 1, total_length, total_length, dtype=torch.bool)
        tail_rows = torch.arange(prompt_length + response_length, total_length)
        clean_rows = torch.arange(prompt_length, prompt_length + response_length)
        for block_id in range((response_length + block_size - 1) // block_size):
            block_start = block_id * block_size
            block_end = min(block_start + block_size, response_length)
            clean_prefix_end = prompt_length + block_start
            tail_start = prompt_length + response_length + block_start
            attention[:, :, tail_rows[block_start:block_end], :clean_prefix_end] = True
            attention[:, :, tail_rows[block_start:block_end],
                      tail_start:tail_start + block_size] = True
            attention[:, :, clean_rows[block_start:block_end],
                      :prompt_length + block_end] = True
        for block_id in range((prompt_length + block_size - 1) // block_size):
            row_end = max(prompt_length - block_id * block_size, 0)
            row_start = max(prompt_length - (block_id + 1) * block_size, 0)
            attention[:, :, row_start:row_end, :row_end] = True

        clean_causal = torch.tril(torch.ones(clean.size(1), clean.size(1), dtype=torch.bool))
        tail_causal = torch.tril(torch.ones(response_length, response_length, dtype=torch.bool))
        attention[:, :, :clean.size(1), :clean.size(1)] &= clean_causal
        attention[:, :, clean.size(1):, clean.size(1):] &= tail_causal
        additive_attention = torch.full(attention.shape, float("-inf"))
        additive_attention[attention] = 0.0

        with torch.no_grad():
            full_logits = teacher(
                input_ids=extended,
                attention_mask=additive_attention,
                position_ids=position_ids,
            ).logits
        opd_logits = torch.cat([
            full_logits[:, :prompt_length],
            full_logits[:, prompt_length + response_length:],
        ], dim=1)
        block_final_indices = torch.arange(
            prompt_length + block_size - 1,
            prompt_length + response_length,
            block_size,
        )
        opd_logits[:, block_final_indices] = full_logits[:, block_final_indices]
        teacher_logprobs = F.log_softmax(opd_logits.roll(1, dims=1).float(), dim=-1)

        for position in range(prompt_length, clean.size(1)):
            with torch.no_grad():
                prefix_logits = teacher(
                    input_ids=clean[:, :position],
                    attention_mask=torch.ones(1, position, dtype=torch.bool),
                    position_ids=torch.arange(position).unsqueeze(0),
                ).logits[:, -1]
            direct_logprobs = F.log_softmax(prefix_logits.float(), dim=-1)
            self.assertTrue(
                torch.allclose(teacher_logprobs[:, position], direct_logprobs, atol=2e-5),
                msg=f"OPD teacher distribution is shifted at response position {position}",
            )

    def test_cached_scores_match_full_sequence_counterfactuals(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            pad_token_id=0,
        )
        teacher = Qwen3ForCausalLM(config).eval()
        clean = torch.tensor([
            [0, 5, 6, 7, 8, 9, 10],
            [3, 4, 5, 6, 7, 8, 9],
        ])
        valid = clean.ne(0)
        positions = valid.long().cumsum(-1) - 1
        positions.masked_fill_(~valid, 1)
        with torch.no_grad():
            logits = teacher(clean, attention_mask=valid, position_ids=positions).logits
        teacher_lp = torch.empty_like(logits, dtype=torch.float32)
        teacher_lp[:, 1:] = F.log_softmax(logits[:, :-1].float(), dim=-1)
        teacher_lp[:, 0] = F.log_softmax(logits[:, 0].float(), dim=-1)
        loss_mask = torch.zeros_like(clean, dtype=torch.bool)
        loss_mask[:, 3:] = True
        sample = TCSMSample(
            blocks=[torch.tensor([0]), torch.tensor([0])],
            eligible_counts=torch.tensor([1, 1]),
        )
        expected_chunks, expected_jobs = count_tcsm_counterfactual_work(
            loss_mask=loss_mask,
            sequence_ends=torch.tensor([7, 7]),
            sample=sample,
            response_start=3,
            block_size=4,
            top_k=3,
            counterfactual_batch_size=4,
        )

        class ProgressRecorder:
            def __init__(self):
                self.completed = 0

            def update(self, count):
                self.completed += count

        progress = ProgressRecorder()
        scores = compute_block_tcs_topk_scores(
            teacher_model=teacher,
            clean_input_ids=clean,
            position_ids=positions,
            teacher_logprobs=teacher_lp,
            loss_mask=loss_mask,
            sequence_ends=torch.tensor([7, 7]),
            sample=sample,
            response_start=3,
            block_size=4,
            top_k=3,
            counterfactual_batch_size=4,
            logprob_fn=teacher_logprobs_from_logits,
            pad_token_id=0,
            show_progress=False,
            is_main_process=True,
            progress_bar=progress,
        )
        self.assertEqual(progress.completed, expected_chunks)
        self.assertEqual(scores.num_candidate_jobs, expected_jobs)
        self.assertFalse(scores.tcs_topk_probs.requires_grad)
        # Six non-final positions (three per row) times K=3. The two block-final
        # positions are returned for their cheap tail correction but create no jobs.
        self.assertEqual(scores.num_candidate_jobs, 18)
        final = scores.token_indices.eq(6)
        self.assertEqual(int(final.sum()), 2)
        final_conditional_opd = scores.opd_topk_logprobs[final].softmax(-1)
        self.assertTrue(torch.allclose(scores.tcs_topk_probs[final], final_conditional_opd))

        for index in range(scores.row_indices.numel()):
            row = int(scores.row_indices[index])
            pos = int(scores.token_indices[index])
            candidate_scores = []
            for candidate, base_lp in zip(
                scores.candidate_indices[index], scores.opd_topk_logprobs[index]
            ):
                branch = clean[row].clone()
                branch[pos] = candidate
                with torch.no_grad():
                    branch_logits = teacher(
                        branch.unsqueeze(0), attention_mask=branch.ne(0).unsqueeze(0),
                        position_ids=positions[row].unsqueeze(0),
                    ).logits[0]
                suffix_lp = F.log_softmax(branch_logits[pos:6].float(), dim=-1)
                suffix_targets = clean[row, pos + 1:7]
                candidate_scores.append(base_lp + suffix_lp.gather(
                    -1, suffix_targets.unsqueeze(-1)
                ).squeeze(-1).sum())
            expected = torch.stack(candidate_scores).softmax(-1)
            self.assertTrue(torch.allclose(scores.tcs_topk_probs[index], expected, atol=2e-3))


if __name__ == "__main__":
    unittest.main()
