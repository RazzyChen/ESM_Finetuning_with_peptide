import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
import argparse
import os
from typing import List, Dict, Any
import warnings
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def analyze_sequences(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze sequences in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'sequence' column.

    Returns:
        Dict[str, Any]: Dictionary containing sequence analysis results.
    """
    sequences = df['sequence'].tolist()
    lengths = [len(seq) for seq in sequences]
    return {
        'count': len(sequences),
        'lengths': lengths,
        'mean_length': np.mean(lengths),
        'variance': np.var(lengths)
    }

def plot_length_histogram(lengths: List[int], title: str) -> None:
    """
    Plot histogram of sequence lengths and save as PNG.

    Args:
        lengths (List[int]): List of sequence lengths.
        title (str): Title for the plot and filename.
    """
    #plt.figure(figsize=(10, 8))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title(title)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def count_tokens_esm2(sequences: List[str]) -> int:
    """
     Count total number of tokens using ESM2 tokenizer.

     Args:
         sequences (List[str]): List of sequences to tokenize.

     Returns:
         int: Total number of tokens.
     """
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    tokens = tokenizer(sequences, return_length=True)
    return sum(tokens['length']) - (2 * len(sequences)) 

def main(csv_files: List[str]) -> None:
    """
    Main function to process CSV files and analyze sequences.

    Args:
        csv_files (List[str]): List of CSV file paths to analyze.
    """
    print("Analyzing...")
    if len(csv_files) == 1:
        # Single file processing
        print(f"\nAnalyzing Dataset: {csv_files[0]}")
        df = load_data(csv_files[0])
        
        stats = analyze_sequences(df)
        tokens = count_tokens_esm2(df['sequence'].tolist())
        
        print("Dataset statistics:")
        print(f"Number of sequences: {stats['count']}")
        print(f"Mean sequence length: {stats['mean_length']:.2f}")
        print(f"Variance of sequence length: {stats['variance']:.2f}")
        print(f"Total number of tokens in the dataset (Using ESM2 tokenizer): {tokens}")
        
        plot_length_histogram(stats['lengths'], "Sequence Length Distribution")
        
        print(f"\nTotal number of tokens (Using ESM2 tokenizer): {tokens}")
    
    else:
        # Multiple files processing
        all_data = pd.DataFrame()
        
        for i, file in enumerate(csv_files, 1):
            print(f"\nAnalyzing Dataset: {file}")
            df = load_data(file)
            
            stats = analyze_sequences(df)
            tokens = count_tokens_esm2(df['sequence'].tolist())
            
            print(f"Dataset statistics:")
            print(f"Number of sequences: {stats['count']}")
            print(f"Mean sequence length: {stats['mean_length']:.2f}")
            print(f"Variance of sequence length: {stats['variance']:.2f}")
            print(f"Total number of tokens in the dataset (Using ESM2 tokenizer): {tokens}")
            
            plot_length_histogram(stats['lengths'], f"Dataset {i} Sequence Length Distribution")
            
            df['dataset'] = f'Dataset {i}'
            all_data = pd.concat([all_data, df], ignore_index=True)
        
        # Save combined data
        combined_file = 'combined_datasets.csv'
        all_data.to_csv(combined_file, index=False)
        print(f"\nCombined data saved to {combined_file}")
        
        # Analyze combined data
        combined_tokens = count_tokens_esm2(all_data['sequence'].tolist())
        print(f"\nTotal number of tokens in combined dataset (Using ESM2 tokenizer): {combined_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze one or multiple sequence datasets.")
    parser.add_argument('csv_files', nargs='+', help='CSV file(s) to analyze')
    args = parser.parse_args()
    
    main(args.csv_files)