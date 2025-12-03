# LangSmith Evaluation

from langsmith import Client
from langsmith.evaluation import evaluate
from config import load_langsmith_api_key
from graph import app
from test_questions import TEST_QUESTIONS
import datetime

# Initialize LangSmith client
load_langsmith_api_key()
client = Client()

# Pull your custom evaluator from LangSmith
evaluator_prompt = client.pull_prompt("eval_research_evaluation_insight_hallucination_evaluator_7349d2f9", include_model=True)


def create_dataset_from_questions(dataset_name=None):
    """
    Create a LangSmith dataset from the 50 test questions
    """
    if dataset_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"statistical-learning-batch-{timestamp}"

    print(f"Creating dataset: {dataset_name}")

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=f"Batch evaluation with {len(TEST_QUESTIONS)} questions about Statistical Learning"
    )

    # Add all questions as examples
    for i, question in enumerate(TEST_QUESTIONS, 1):
        client.create_example(
            inputs={"query": question, "answer": "", "source": ""},
            outputs={},  # No reference outputs needed
            dataset_id=dataset.id
        )
        print(f"Added question {i}/{len(TEST_QUESTIONS)}")

    print(f"✅ Dataset '{dataset_name}' created with {len(TEST_QUESTIONS)} examples")
    return dataset_name


def run_batch_evaluation(dataset_name=None):
    """
    Run evaluation on all 50 questions using your LangSmith evaluator
    """
    if dataset_name is None:
        # Create new dataset
        dataset_name = create_dataset_from_questions()

    print(f"\n🚀 Starting batch evaluation on dataset: {dataset_name}")
    print(f"📊 Total questions: {len(TEST_QUESTIONS)}")

    # Define prediction function
    def predict(inputs):
        """Run the RAG system for each question"""
        result = app.invoke(inputs)
        return result

    # Run evaluation using your custom LangSmith evaluator
    # The evaluator you created on LangSmith UI will be automatically applied
    results = evaluate(
        predict,
        data=dataset_name,
        experiment_prefix="statistical-learning-batch-eval",
        metadata={
            "model": "gpt-4o",
            "num_questions": len(TEST_QUESTIONS),
            "evaluator": "insight_hallucination_evaluator_7349d2f9"
        }
    )

    print("\n✅ Batch evaluation complete!")
    print(f"📈 View detailed results at: https://smith.langchain.com/")
    print(f"📊 Project: {dataset_name}")

    return results


def get_evaluation_statistics(experiment_name):
    """
    Get statistics from completed evaluation runs
    """
    print(f"\n📊 Fetching evaluation statistics for: {experiment_name}")

    # Get runs from the experiment
    runs = list(client.list_runs(project_name=experiment_name))

    total_runs = len(runs)
    hallucination_scores = []

    for run in runs:
        if run.feedback_stats:
            # Extract hallucination scores from feedback
            for feedback in run.feedback_stats:
                if 'hallucination' in feedback.key.lower():
                    hallucination_scores.append(feedback.score)

    if hallucination_scores:
        avg_hallucination = sum(hallucination_scores) / len(hallucination_scores)
        hallucination_rate = avg_hallucination  # If 1.0 = hallucinated, 0.0 = grounded

        print(f"\n📈 Evaluation Statistics:")
        print(f"   Total Runs: {total_runs}")
        print(f"   Average Hallucination Score: {avg_hallucination:.3f}")
        print(f"   Hallucination Rate: {hallucination_rate * 100:.2f}%")
        print(f"   Grounded Rate: {(1 - hallucination_rate) * 100:.2f}%")

        return {
            "total_runs": total_runs,
            "avg_hallucination_score": avg_hallucination,
            "hallucination_rate": hallucination_rate
        }
    else:
        print("⚠️ No hallucination scores found. Make sure evaluator is properly configured.")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "create-dataset":
            # Create dataset only
            dataset_name = sys.argv[2] if len(sys.argv) > 2 else None
            create_dataset_from_questions(dataset_name)

        elif sys.argv[1] == "run":
            # Run full evaluation
            dataset_name = sys.argv[2] if len(sys.argv) > 2 else None
            run_batch_evaluation(dataset_name)

        elif sys.argv[1] == "stats":
            # Get statistics from existing experiment
            if len(sys.argv) > 2:
                get_evaluation_statistics(sys.argv[2])
            else:
                print("Please provide experiment name: python evaluation.py stats <experiment_name>")
    else:
        print("Usage:")
        print("  python evaluation.py create-dataset [dataset_name]  # Create dataset with 50 questions")
        print("  python evaluation.py run [dataset_name]             # Run batch evaluation")
        print("  python evaluation.py stats <experiment_name>        # Get statistics")
        print("\nExample:")
        print("  python evaluation.py run                            # Creates dataset and runs evaluation")
        print("  python evaluation.py stats statistical-learning-batch-eval")  # View results