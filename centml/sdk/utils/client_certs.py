import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import click
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


@dataclass
class CAClientCertTriplet:
    certificate_authority: str
    client_certificate: str
    client_private_key: str


# Generate the client certificate for private endpoint
def generate_ca_client_triplet(service_name: str) -> CAClientCertTriplet:
    # Generate private key for CA using ECDSA
    ca_private_key = ec.generate_private_key(ec.SECP384R1())

    # Details about who we are. For a self-signed certificate, the subject
    # and issuer are always the same.
    ca_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"ca.{service_name}.user.centml.ai")])

    ca_certificate = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        # Certificate valid for 5 years (give or take leap years)
        .not_valid_after(datetime.utcnow() + timedelta(days=365 * 5))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        # We are using SHA384 as it's often paired with secpr384r1, ie
        # the weak link isn't the hash algorithm.
        .sign(private_key=ca_private_key, algorithm=hashes.SHA384())
    )

    # Generate private key for Client using ECDSA
    client_private_key = ec.generate_private_key(ec.SECP384R1())

    # Information about the client
    client_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"client.{service_name}.user.centml.ai")])

    client_certificate = (
        x509.CertificateBuilder()
        .subject_name(client_subject)
        # The issuer is the CA that we made earlier
        .issuer_name(ca_certificate.subject)
        .public_key(client_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        # Certificate valid for 5 years (give or take leap years)
        .not_valid_after(datetime.utcnow() + timedelta(days=365 * 5))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        # We are using SHA384 as it's often paired with secpr384r1, ie
        # the weak link isn't the hash algorithm.
        .sign(ca_private_key, hashes.SHA384())
    )

    return CAClientCertTriplet(
        certificate_authority=ca_certificate.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        client_certificate=client_certificate.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        client_private_key=client_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("ascii"),
    )


# Util to automatically save the private endpoint in a combined pem format for the user
def save_pem_file(service_name, client_private_key, client_certificate):
    """Generate a PEM file and save it to the current directory for private endpoint."""

    # Get the current working directory
    current_directory = os.getcwd()

    # Define the file path
    ca_file_path = os.path.join(current_directory, f"{service_name}.pem")

    try:
        # Save the combined PEM file
        with open(ca_file_path, 'w') as combined_pem_file:
            combined_pem_file.write(client_private_key + client_certificate)
        click.echo(f"Combined PEM file for accessing the private endpoint has been saved to {ca_file_path}")

    except Exception as e:
        click.echo(f"Error saving PEM files: {e}")
