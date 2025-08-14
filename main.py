#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Product Classifier - Model Training
=====================================

This script trains a neural network for automatic product classification
based on their names. The system uses modern machine learning methods:
- Sentence Transformers for understanding the meaning of names
- Neural network for classification
- Strategies to prevent overfitting

Author: Nikolai Kulian
Date: 2025
"""

# ============================================================================
# LIBRARY IMPORTS
# ============================================================================

# Standard Python libraries
import os
import pickle
import time

# Data processing libraries
import numpy as np
import pandas as pd

# Machine learning libraries
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Neural network libraries
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical

# Text understanding library
from sentence_transformers import SentenceTransformer

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log(message):
    """
    Beautifully displays messages with timestamps
    
    Args:
        message (str): Message to display
    """
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def load_and_prepare_data():
    """
    Loads and prepares data for training
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    log("Loading product dataset...")
    
    # Load CSV file with products
    # sep=";" - column separator
    # encoding="utf-8" - encoding for Russian text
    # on_bad_lines="skip" - skip lines with errors
    df = pd.read_csv("products.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
    
    # Keep only needed columns and remove empty rows
    df = df[['name', 'fullGroupName']].dropna()
    
    log(f"Loaded {len(df)} products")
    
    # ========================================================================
    # CATEGORY PREPARATION
    # ========================================================================
    
    log("Preparing product categories...")
    
    # Create category encoder
    # LabelEncoder converts text category names to numbers
    label_encoder = LabelEncoder()
    
    # Encode categories (convert text to numbers)
    df['category_encoded'] = label_encoder.fit_transform(df['fullGroupName'])
    
    # Count how many products in each category
    category_counts = df['category_encoded'].value_counts()
    
    # Keep only categories with 3+ products (remove rare ones)
    valid_categories = category_counts[category_counts >= 3].index
    df = df[df['category_encoded'].isin(valid_categories)]
    
    log(f"Remaining {len(df)} products in {len(valid_categories)} categories")
    
    # ========================================================================
    # TRAIN/TEST SPLIT
    # ========================================================================
    
    log("Splitting data into training and testing sets...")
    
    # Split data: 80% for training, 20% for testing
    # stratify - preserve category proportions in both parts
    X_train_texts, X_test_texts, y_train_raw, y_test_raw = train_test_split(
        df['name'],           # product names
        df['fullGroupName'],  # product categories
        test_size=0.2,        # 20% for testing
        random_state=42,      # fix randomness for reproducibility
        stratify=df['category_encoded']  # preserve category proportions
    )
    
    log(f"Training set: {len(X_train_texts)} products")
    log(f"Test set: {len(X_test_texts)} products")
    
    return X_train_texts, X_test_texts, y_train_raw, y_test_raw, label_encoder

def create_text_embeddings(X_train_texts, X_test_texts):
    """
    Converts product names to numerical vectors (embeddings)
    
    Args:
        X_train_texts: product names for training
        X_test_texts: product names for testing
    
    Returns:
        tuple: (X_train_vectors, X_test_vectors)
    """
    log("Loading model for understanding text meaning...")
    
    # Load pre-trained model for understanding text meaning
    # This model can work with Russian and English languages
    transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Check if ready embeddings already exist
    train_vectors_file = "X_train_vectors.npy"
    test_vectors_file = "X_test_vectors.npy"
    
    if os.path.exists(train_vectors_file) and os.path.exists(test_vectors_file):
        log("Loading ready embeddings from files...")
        X_train_vectors = np.load(train_vectors_file)
        X_test_vectors = np.load(test_vectors_file)
    else:
        log("Creating embeddings for product names...")
        
        # Convert names to vectors
        # batch_size=32 - process 32 names at a time
        # show_progress_bar=True - show progress
        X_train_vectors = transformer.encode(
            X_train_texts.tolist(), 
            batch_size=32, 
            show_progress_bar=True
        )
        
        X_test_vectors = transformer.encode(
            X_test_texts.tolist(), 
            batch_size=32, 
            show_progress_bar=True
        )
        
        # Save embeddings for reuse
        log("Saving embeddings to files...")
        np.save(train_vectors_file, X_train_vectors)
        np.save(test_vectors_file, X_test_vectors)
    
    log(f"Embedding size: {X_train_vectors.shape[1]} numbers")
    return X_train_vectors, X_test_vectors

def prepare_labels(y_train_raw, y_test_raw, X_test_vectors):
    """
    Prepares labels (categories) for training
    
    Args:
        y_train_raw: raw categories for training
        y_test_raw: raw categories for testing
        X_test_vectors: test data vectors
    
    Returns:
        tuple: (y_train, y_test, y_train_categorical, y_test_categorical, label_encoder)
    """
    log("Preparing labels for training...")
    
    # Create new encoder only for training data
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw)
    
    # Encode categories to numbers
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw[y_test_raw.isin(label_encoder.classes_)])
    
    # Filter test vectors under filtered labels
    test_mask = y_test_raw.isin(label_encoder.classes_).to_numpy()
    X_test_filtered = X_test_vectors[test_mask]
    
    # Check that we have test data
    if len(y_test) == 0:
        raise ValueError("No test data left after filtering!")
    
    # Convert labels to "one-hot encoding" format
    # Example: category 2 out of 5 possible = [0, 0, 1, 0, 0]
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)
    
    log(f"Prepared {len(y_train)} training and {len(y_test)} test examples")
    log(f"Number of categories: {y_train_categorical.shape[1]}")
    
    return y_train, y_test, y_train_categorical, y_test_categorical, label_encoder, X_test_filtered

def create_neural_network(input_size, output_size):
    """
    Creates neural network for classification
    
    Args:
        input_size: input data size (embedding size)
        output_size: number of categories
    
    Returns:
        Sequential: ready neural network
    """
    log("Creating neural network...")
    
    # Create sequential model (layers go one after another)
    model = Sequential([
        # First layer: accepts embeddings (384 numbers)
        Dense(
            units=384,                    # 384 neurons
            activation='relu',            # ReLU activation function
            input_shape=(input_size,),    # input data size
            kernel_regularizer=regularizers.l2(0.0001)  # L2 regularization
        ),
        
        # Batch normalization (stabilizes training)
        BatchNormalization(),
        
        # Dropout 25% (randomly disables 25% neurons to prevent overfitting)
        Dropout(0.25),
        
        # Second hidden layer: 192 neurons
        Dense(
            units=192,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        
        # Dropout 15%
        Dropout(0.15),
        
        # Output layer: number of categories
        Dense(
            units=output_size,
            activation='softmax'  # softmax gives probabilities for each category
        )
    ])
    
    log("Neural network created!")
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Trains neural network
    
    Args:
        model: neural network
        X_train: training data
        y_train: training labels
        X_test: test data
        y_test: test labels
    
    Returns:
        tuple: (trained_model, training_time)
    """
    log("Starting neural network training...")
    
    # Configure model for training
    model.compile(
        optimizer=AdamW(learning_rate=0.0005),  # optimizer with learning rate
        loss='categorical_crossentropy',         # loss function for classification
        metrics=['accuracy']                     # metric - accuracy
    )
    
    # Create "callbacks" - functions that are called during training
    
    # Early Stopping: stop training if accuracy doesn't grow
    early_stopping = EarlyStopping(
        monitor='val_loss',        # monitor validation loss
        patience=5,                # wait 5 epochs
        restore_best_weights=True  # restore best weights
    )
    
    # Reduce Learning Rate: reduce learning rate if accuracy doesn't grow
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',    # monitor validation loss
        factor=0.5,            # reduce by 2 times
        patience=2,            # wait 2 epochs
        min_lr=1e-5,          # minimum learning rate
        verbose=1              # show messages
    )
    
    # Mark training start time
    start_time = time.time()
    
    # Train model
    history = model.fit(
        X_train, y_train,           # training data
        validation_split=0.1,        # 10% data for validation
        epochs=50,                   # maximum 50 epochs
        batch_size=128,              # batch size
        callbacks=[early_stopping, reduce_lr],  # our callbacks
        verbose=1                    # show progress
    )
    
    # Calculate training time
    training_time = round(time.time() - start_time, 2)
    
    log(f"Training completed in {training_time} seconds")
    return model, training_time

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates quality of trained model
    
    Args:
        model: trained neural network
        X_test: test data
        y_test: test labels
        label_encoder: category encoder
    
    Returns:
        tuple: (accuracy, error_rate)
    """
    log("Evaluating model quality...")
    
    # Make predictions on test data
    predictions_probabilities = model.predict(X_test, verbose=0)
    
    # Take category with maximum probability
    predictions = np.argmax(predictions_probabilities, axis=1)
    
    # Calculate accuracy
    accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
    error_rate = round(100 - accuracy, 2)
    
    # Display detailed classification report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    # Get category names for report
    unique_labels = np.unique(y_test)
    category_names = [str(name) for name in label_encoder.inverse_transform(unique_labels)]
    
    # Display sklearn report
    print(classification_report(y_test, predictions, 
                              labels=unique_labels, 
                              target_names=category_names))
    
    # Display final results
    print("="*60)
    print(f"ACCURACY: {accuracy}%")
    print(f"ERROR RATE: {error_rate}%")
    print("="*60)
    
    return accuracy, error_rate

def save_model_and_encoder(model, label_encoder):
    """
    Saves trained model and category encoder
    
    Args:
        model: trained neural network
        label_encoder: category encoder
    """
    log("Saving model and encoder...")
    
    # Save neural network
    model.save("class_model.h5")
    log("Model saved as 'class_model.h5'")
    
    # Save category encoder
    with open("label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)
    log("Category encoder saved as 'label_encoder.pkl'")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function that performs the entire training process
    """
    try:
        log("STARTING PRODUCT CLASSIFICATION MODEL TRAINING")
        log("=" * 60)
        
        # Step 1: Load and prepare data
        X_train_texts, X_test_texts, y_train_raw, y_test_raw, label_encoder = load_and_prepare_data()
        
        # Step 2: Create embeddings (vectors) for product names
        X_train_vectors, X_test_vectors = create_text_embeddings(X_train_texts, X_test_texts)
        
        # Step 3: Prepare labels for training
        y_train, y_test, y_train_categorical, y_test_categorical, label_encoder, X_test_filtered = prepare_labels(
            y_train_raw, y_test_raw, X_test_vectors
        )
        
        # Step 4: Create neural network
        model = create_neural_network(
            input_size=X_train_vectors.shape[1],
            output_size=y_train_categorical.shape[1]
        )
        
        # Step 5: Train model
        trained_model, training_time = train_model(
            model, X_train_vectors, y_train_categorical, 
            X_test_filtered, y_test_categorical
        )
        
        # Step 6: Evaluate quality
        accuracy, error_rate = evaluate_model(
            trained_model, X_test_filtered, y_test, label_encoder
        )
        
        # Step 7: Save results
        save_model_and_encoder(trained_model, label_encoder)
        
        # Final report
        log("TRAINING COMPLETED SUCCESSFULLY!")
        log(f"Model accuracy: {accuracy}%")
        log(f"Training time: {training_time} seconds")
        log(f"Number of categories: {y_train_categorical.shape[1]}")
        log("Model is ready for use!")
        
    except Exception as error:
        log(f"ERROR: {error}")
        log("Check data and try again")

# ============================================================================
# PROGRAM EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run main function only if script is run directly
    main()