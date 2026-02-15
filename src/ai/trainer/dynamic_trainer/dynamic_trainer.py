from ..trainer import Trainer
from ai.config import config
import logging


class DynamicTrainer(Trainer):
    def _check_graduation(self, level_idx: int) -> bool:
        """
        Decides if the agent is ready for the next level.
        Order of operations:
        1. Max Episodes (Fail-safe)
        2. Training Metrics (Consistency + Plateau)
        3. Real Exam (Determininstic Validation)
        """
        current_episodes = self.metrics.num_episodes

        # Fail-safe
        if current_episodes >= config.DYNAMIC_MAX_EPISODES:
            logging.warning(
                f"🛑 Level {level_idx+1} Timeout (Max Episodes). Moving on."
            )
            return True

        # Check Training Metrics
        success_rate = self.metrics.get_binary_success_rate()
        target_consistency = 0.80

        is_consistent = success_rate >= target_consistency
        is_plateaued = self.metrics.is_plateaued(
            min_episodes=config.DYNAMIC_MIN_EPISODES
        )

        # Log status periodically
        if current_episodes % 50 == 0:
            logging.info(
                f"📊 Dynamic Check: Success={success_rate:.1%} (Target {target_consistency:.0%}), "
                f"Plateaued={is_plateaued}"
            )

        # Only proceed to the expensive Exam if training looks promising
        if is_consistent and is_plateaued:
            logging.info("🎓 Training looks good. Administering Competence Exam...")

            # The Real Exam
            exam_score = self.evaluator.evaluate_competence(
                self.agent_manager.get_policy(), num_episodes=10
            )

            if exam_score >= config.DYNAMIC_EXAM_PASS_REWARD:
                logging.info(
                    f"🏆 PASSED! Validation Score: {exam_score:.1%}. Graduating."
                )
                return True
            else:
                logging.warning(
                    f"❌ FAILED EXAM. Training says {success_rate:.1%}, "
                    f"but Reality (Greedy) says {exam_score:.1%}. Continuing training..."
                )
                return False

        return False
