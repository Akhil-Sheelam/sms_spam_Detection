#Import necessary libraries
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Script started...", flush=True)

#Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')  # Load CSV file
df = df[['v1', 'v2']]                             # Keep only useful columns
df.columns = ['label', 'text']                    # Rename columns

#Encode the labels ('ham' = 0, 'spam' = 1)
df['label'] = LabelEncoder().fit_transform(df['label'])

#Prepare the text data
texts = df['text'].values                         # Input texts
labels = df['label'].values                       # Output labels

#Tokenize and pad the texts
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post')

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

#Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16, input_length=padded.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model with visual output
print(" Training the model...", flush=True)
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=2
)

#Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n Final Test Accuracy: {accuracy:.4f}", flush=True)

#Plot training and validation accuracy/loss
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)