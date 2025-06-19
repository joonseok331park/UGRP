import json
import pandas as pd
from itertools import chain

class CANTokenizer:
    """
    Manages the vocabulary for CAN data and handles token-to-index conversion.
    Implements an offset-based vocabulary as described in Jo & Kim (2024).
    """
    def __init__(self):
        """
        Initializes the tokenizer, including special tokens and the ID offset.
        <PAD> token is set to index 0.
        """
        self.token_to_id = {}
        self.id_to_token = {}
        self.ID_OFFSET = 260  # Offset for CAN IDs to avoid collision with data tokens

        # Add special tokens. <PAD> must be at index 0.
        self.special_tokens = ['<PAD>', '<UNK>', '<MASK>', '<VOID>']
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token):
        """Helper to add a token to the vocabulary."""
        if token not in self.token_to_id:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

    def build_vocab(self, df: pd.DataFrame) -> None:
        """
        Builds an offset-based integrated vocabulary from a DataFrame.
        1. Data Tokens: '00' to 'FF' (256 tokens).
        2. ID Tokens: Unique CAN IDs from the DataFrame, offset by self.ID_OFFSET.
        """
        # 1. Add data tokens ('00' to 'FF')
        for i in range(256):
            self._add_token(f'{i:02X}')

        # 2. Add ID tokens with an offset
        unique_ids = df['CAN_ID'].unique()
        for can_id in unique_ids:
            # The token is the string representation of the integer value (ID + offset)
            token = str(int(can_id, 16) + self.ID_OFFSET)
            self._add_token(token)

    def save_vocab(self, file_path: str) -> None:
        """Saves the token_to_id dictionary to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=4)

    def load_vocab(self, file_path: str) -> None:
        """Loads a vocabulary from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            # All keys are strings in JSON, which now matches our vocabulary structure.
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, tokens: list) -> list[int]:
        """Converts a list of tokens to a list of integer IDs."""
        unk_token_id = self.token_to_id['<UNK>']
        return [self.token_to_id.get(token, unk_token_id) for token in tokens]

    def decode(self, ids: list[int]) -> list:
        """Converts a list of integer IDs back to a list of tokens."""
        return [self.id_to_token.get(id, '<UNK>') for id in ids]


class CANSequencer:
    """
    Transforms a DataFrame of CAN messages into fixed-length sequences
    using a CANTokenizer.
    """
    def __init__(self, tokenizer: CANTokenizer, seq_len: int = 126):
        """
        Initializes the sequencer.
        :param tokenizer: An instance of CANTokenizer.
        :param seq_len: The fixed length of each sequence.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def transform(self, df: pd.DataFrame) -> list[list[int]]:
        """
        Transforms the DataFrame into a list of encoded sequences.
        This method now expects the 'Data' column to be a list of hex strings.
        """
        # 1. Frame-level tokenization
        all_tokens = []
        for _, row in df.iterrows():
            # Use the offset from the tokenizer for consistency
            can_id_token = str(int(row['CAN_ID'], 16) + self.tokenizer.ID_OFFSET)
            
            # Directly use the 'Data' column, which is expected to be a list of strings
            data_bytes = row['Data']
            
            # Each frame is a list of 9 tokens (ID + 8 data bytes)
            frame_tokens = [can_id_token] + data_bytes
            all_tokens.append(frame_tokens)

        # 2. Create a single stream
        token_stream = list(chain.from_iterable(all_tokens))

        # 3. Sliding window
        sequences = []
        stride = 1
        for i in range(0, len(token_stream) - self.seq_len + 1, stride):
            sequence = token_stream[i:i + self.seq_len]
            sequences.append(sequence)

        # 4. Encoding
        encoded_sequences = [self.tokenizer.encode(seq) for seq in sequences]

        return encoded_sequences


if __name__ == '__main__':
    # 1. Create a sample DataFrame
    # 1. Create a sample DataFrame matching the data_loader.py output format
    sample_data = {
        'CAN_ID': ['01A', '02B', '01A', '03C'],
        'Data': [
            ['11', '22', '33', '44', '55', '66', '77', '88'],
            ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', '00', '11'],
            ['99', '88', '77', '66', '55', '44', '33', '22'],
            ['DE', 'AD', 'BE', 'EF', 'CA', 'FE', 'BA', 'BE']
        ]
    }
    df_sample = pd.DataFrame(sample_data)

    print("--- CAN Tokenizer and Sequencer Test ---")
    print("\n1. Sample DataFrame:")
    print(df_sample)

    # 2. Initialize tokenizer and build vocabulary
    tokenizer = CANTokenizer()
    tokenizer.build_vocab(df_sample)
    print("\n2. Vocabulary built successfully.")

    # 3. Check vocabulary content
    print("\n3. Vocabulary Snippet:")
    print("  - Special Tokens:", {k: v for k, v in tokenizer.token_to_id.items() if k in tokenizer.special_tokens})
    print("  - Data Tokens (sample):", {k: v for i, (k, v) in enumerate(tokenizer.token_to_id.items()) if i >= 4 and i < 8})
    id_tokens_to_check = {str(int(id, 16) + tokenizer.ID_OFFSET) for id in df_sample['CAN_ID'].unique()}
    print(f"  - ID Tokens (offset {tokenizer.ID_OFFSET}):", {k: v for k, v in tokenizer.token_to_id.items() if k in id_tokens_to_check})


    # 4. Test saving and loading vocabulary
    vocab_file = 'vocab.json'
    tokenizer.save_vocab(vocab_file)
    print(f"\n4. Vocabulary saved to '{vocab_file}'.")
    new_tokenizer = CANTokenizer()
    new_tokenizer.load_vocab(vocab_file)
    print(f"   Vocabulary loaded from '{vocab_file}' successfully.")
    # Verify loaded vocab
    assert tokenizer.token_to_id == new_tokenizer.token_to_id
    print("   Loaded vocabulary matches original vocabulary.")


    # 5. Initialize sequencer
    sequence_length = 10
    sequencer = CANSequencer(tokenizer=tokenizer, seq_len=sequence_length)
    print(f"\n5. CANSequencer initialized with seq_len={sequence_length}.")

    # 6. Transform data
    encoded_sequences = sequencer.transform(df_sample)
    print("\n6. DataFrame transformed into encoded sequences.")

    # 7. Print and verify results
    print("\n--- Verification ---")
    if encoded_sequences:
        first_sequence_encoded = encoded_sequences[0]
        first_sequence_decoded = tokenizer.decode(first_sequence_encoded)

        print("\nInput DataFrame Head:")
        print(df_sample.head(2))
        print("\nFirst Generated Encoded Sequence (len={}):".format(len(first_sequence_encoded)))
        print(first_sequence_encoded)
        print("\nFirst Sequence Decoded Back to Tokens (len={}):".format(len(first_sequence_decoded)))
        print(first_sequence_decoded)

        # Manually verify the first few tokens
        print("\nManual Verification of First Decoded Sequence:")
        id_01A_token = str(int('01A', 16) + tokenizer.ID_OFFSET)
        print(f"  - Expected first token: '{id_01A_token}' (ID: 01A + offset {tokenizer.ID_OFFSET})")
        print(f"  - Actual first token:   '{first_sequence_decoded[0]}'")
        print(f"  - Expected second token: '11'")
        print(f"  - Actual second token:   '{first_sequence_decoded[1]}'")
        assert first_sequence_decoded[0] == id_01A_token
        assert first_sequence_decoded[1] == '11'
        print("\nVerification successful!")
    else:
        print("\nNo sequences were generated. The token stream might be shorter than seq_len.")