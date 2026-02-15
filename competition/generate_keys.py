from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import os

def generate_keys():
    # generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # serialize private key
    pem_private = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # serialize public key
    public_key = private_key.public_key()
    pem_public = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # save keys
    os.makedirs('competition', exist_ok=True)
    
    with open('competition/submission_key', 'wb') as f:
        f.write(pem_private)
        
    with open('competition/submission_key.pub', 'wb') as f:
        f.write(pem_public)
        
    print("Keys generated successfully!")
    print(f"Private key saved to: {os.path.abspath('competition/submission_key')}")
    print(f"Public key saved to: {os.path.abspath('competition/submission_key.pub')}")
    print("\nIMPORTANT: Add the content of 'competition/submission_key' to your GitHub Secrets as 'SUBMISSION_PRIVATE_KEY'.")

if __name__ == "__main__":
    generate_keys()
