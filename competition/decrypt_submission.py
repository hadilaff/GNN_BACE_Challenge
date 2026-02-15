from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import sys
import os

def decrypt_file(encrypted_file_path, private_key_pem):
    if not os.path.exists(encrypted_file_path):
        print(f"Error: File {encrypted_file_path} not found.")
        return None

    try:
        # Load the private key
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None
        )

        with open(encrypted_file_path, 'rb') as f:
            data = f.read()

        # Extract the encrypted symmetric key
        key_length = int.from_bytes(data[:4], byteorder='big')
        encrypted_symmetric_key = data[4:4+key_length]
        encrypted_content = data[4+key_length:]

        # Decrypt the symmetric key
        symmetric_key = private_key.decrypt(
            encrypted_symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Decrypt the content
        f = Fernet(symmetric_key)
        decrypted_content = f.decrypt(encrypted_content)
        
        return decrypted_content

    except Exception as e:
        print(f"Decryption failed: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python decrypt_submission.py <encrypted_file.enc>")
        sys.exit(1)
        
    encrypted_file = sys.argv[1]
    private_key_env = os.environ.get('SUBMISSION_PRIVATE_KEY')
    
    if not private_key_env:
        print("Error: SUBMISSION_PRIVATE_KEY environment variable not set.")
        sys.exit(1)
        
    decrypted_data = decrypt_file(encrypted_file, private_key_env)
    
    if decrypted_data:
        # Write to a temporary file or stdout
        # For security, we might want to write to a temp file that the evaluator reads
        # But for now, let's write to a generic 'decrypted_submission.csv' 
        # that is .gitignored or deleted immediately after use
        output_path = 'decrypted_submission.csv'
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        print(f"Successfully decrypted to {output_path}")
    else:
        sys.exit(1)
