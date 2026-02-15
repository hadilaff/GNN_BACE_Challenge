from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
import sys
import os

def encrypt_file(file_path, public_key_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    if not os.path.exists(public_key_path):
        print(f"Error: Public key {public_key_path} not found.")
        print("Please ask the organizers to provide the public key.")
        return

    # Read the file to encrypt
    with open(file_path, 'rb') as f:
        data = f.read()

    # Load the public key
    with open(public_key_path, 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read()
        )

    # Encrypt the data
    # Note: RSA can only encrypt small amounts of data directly (based on key size).
    # For large files, we should use symmetric encryption (e.g., Fernet) and encrypt the symmetric key with RSA.
    # However, given submission CSVs are likely small, we'll try hybrid approach for robustness.
    
    # Actually, let's use a symmetric key (Fernet) for the data, and encrypt that key with RSA.
    from cryptography.fernet import Fernet
    
    # Generate a symmetric key
    symmetric_key = Fernet.generate_key()
    f = Fernet(symmetric_key)
    encrypted_data = f.encrypt(data)
    
    # Encrypt the symmetric key with the public RSA key
    encrypted_symmetric_key = public_key.encrypt(
        symmetric_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Combine encrypted key and encrypted data
    # Format: [Key Length (4 bytes)][Encrypted Key][Encrypted Data]
    key_length = len(encrypted_symmetric_key)
    final_data = key_length.to_bytes(4, byteorder='big') + encrypted_symmetric_key + encrypted_data
    
    output_path = file_path.replace('.csv', '.enc')
    with open(output_path, 'wb') as f:
        f.write(final_data)
        
    print(f"File encrypted successfully: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encrypt_submission.py <submission_file.csv>")
        sys.exit(1)
        
    submission_file = sys.argv[1]
    public_key_file = os.path.join(os.path.dirname(__file__), 'submission_key.pub')
    
    encrypt_file(submission_file, public_key_file)
