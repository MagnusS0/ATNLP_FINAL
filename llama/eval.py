import csv
from dataset_llama import load_data
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from Levenshtein import distance
from collections import Counter
import re
nltk.download('punkt')

def load_predictions(csv_file):
    predictions = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            predictions.append(row['output'])
    return predictions

def calculate_detailed_metrics(predictions, ground_truth, input_texts):
    token_matches = 0
    total_tokens = 0
    sequence_matches = 0
    total_sequences = len(predictions)
    length_diffs = []
    edit_distances = []
    error_examples = []
    
    for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
        pred_tokens = word_tokenize(pred)
        true_tokens = word_tokenize(true)
        
        # Token accuracy
        curr_matches = sum(1 for p_tok, t_tok in zip(pred_tokens, true_tokens) if p_tok == t_tok)
        token_matches += curr_matches
        total_tokens += len(true_tokens)
        
        # Length difference
        length_diffs.append(len(pred_tokens) - len(true_tokens))
        
        # Edit distance
        edit_distances.append(distance(pred, true))
        
        # Store error examples
        if pred.strip() != true.strip():
            error_examples.append((i, pred, true))
        
        # Sequence accuracy
        if pred.strip() == true.strip():
            sequence_matches += 1
    
    metrics = {
        'token_accuracy': token_matches / total_tokens if total_tokens > 0 else 0,
        'sequence_accuracy': sequence_matches / total_sequences if total_sequences > 0 else 0,
        'avg_length_diff': np.mean(length_diffs),
        'avg_edit_distance': np.mean(edit_distances),
        'total_samples': total_sequences,
        'exact_matches': sequence_matches,
        'error_examples': error_examples[:5]  # Show first 5 errors
    }

        # Calculate per-length metrics
    input_length_metrics = {}
    output_length_metrics = {}

    for i, (pred, true, inp) in enumerate(zip(predictions, ground_truth, input_texts)):
        # Input length metrics
        input_len = len(word_tokenize(inp))
        if input_len not in input_length_metrics:
            input_length_metrics[input_len] = {
                'token_matches': 0,
                'total_tokens': 0,
                'sequence_matches': 0,
                'count': 0
            }
        
        # Output length metrics
        output_len = len(word_tokenize(true))
        if output_len not in output_length_metrics:
            output_length_metrics[output_len] = {
                'token_matches': 0,
                'total_tokens': 0,
                'sequence_matches': 0,
                'count': 0
            }
        
        # Calculate metrics for this sample
        pred_tokens = word_tokenize(pred)
        true_tokens = word_tokenize(true)
        curr_matches = sum(1 for p_tok, t_tok in zip(pred_tokens, true_tokens) if p_tok == t_tok)
        
        # Update input length metrics
        input_length_metrics[input_len]['token_matches'] += curr_matches
        input_length_metrics[input_len]['total_tokens'] += len(true_tokens)
        input_length_metrics[input_len]['sequence_matches'] += (pred.strip() == true.strip())
        input_length_metrics[input_len]['count'] += 1
        
        # Update output length metrics
        output_length_metrics[output_len]['token_matches'] += curr_matches
        output_length_metrics[output_len]['total_tokens'] += len(true_tokens)
        output_length_metrics[output_len]['sequence_matches'] += (pred.strip() == true.strip())
        output_length_metrics[output_len]['count'] += 1

    # Calculate final accuracies
    for metrics_dict in [input_length_metrics, output_length_metrics]:
        for length in metrics_dict:
            stats = metrics_dict[length]
            stats['token_accuracy'] = stats['token_matches'] / stats['total_tokens'] if stats['total_tokens'] > 0 else 0
            stats['sequence_accuracy'] = stats['sequence_matches'] / stats['count'] if stats['count'] > 0 else 0
            # Remove intermediate calculation fields
            del stats['token_matches']
            del stats['total_tokens']
            del stats['sequence_matches']

    metrics['input_length_metrics'] = input_length_metrics
    metrics['output_length_metrics'] = output_length_metrics
    
    return metrics

def main():
    predictions = load_predictions('output.csv')
    test_dataset = load_data("./data/simple_split/size_variations/tasks_test_simple_p64.txt", test=True)
    ground_truth = [item['output'] for item in test_dataset]

    # We need to remove the <cmd> and </cmd> tags from input texts use regex
    cleaned_input_texts = [re.sub(r'</?cmd>', '', item['conversations'][1]['content']) for item in test_dataset]

    
    metrics = calculate_detailed_metrics(predictions, ground_truth, cleaned_input_texts)
    
    print("\n=== Evaluation Results ===")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Token Accuracy: {metrics['token_accuracy']:.4f}")
    print(f"Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
    print(f"Exact Matches: {metrics['exact_matches']}")
    print(f"Average Length Difference: {metrics['avg_length_diff']:.2f} tokens")
    print(f"Average Edit Distance: {metrics['avg_edit_distance']:.2f}")
    # print("\n=== Input Length Metrics ===")
    # for length, stats in metrics['input_length_metrics'].items():
    #     print(f"\nLength: {length}")
    #     print(f"Token Accuracy: {stats['token_accuracy']:.4f}")
    #     print(f"Sequence Accuracy: {stats['sequence_accuracy']:.4f}")
    # print("\n=== Output Length Metrics ===")
    # for length, stats in metrics['output_length_metrics'].items():
    #     print(f"\nLength: {length}")
    #     print(f"Token Accuracy: {stats['token_accuracy']:.4f}")
    #     print(f"Sequence Accuracy: {stats['sequence_accuracy']:.4f}")
    
    print("\n=== Error Examples ===")
    for idx, pred, true in metrics['error_examples']:
        print(f"\nExample {idx}:")
        print(f"Predicted: {pred}")
        print(f"True: {true}")
        print(f"Edit Distance: {distance(pred, true)}")

if __name__ == "__main__":
    main()