"""
Generate the local CA and the certificates signed by it.

  certs/ca.crt, ca.key          local CA (ca.key never leaves the host)
  certs/server.crt, server.key  NGINX front door (SAN: localhost, nginx, 127.0.0.1)
  certs/mitm.pem                mitmproxy listener, key + cert in one PEM (SAN: mitm)

Only missing files are generated, so it is safe to re-run. Set FORCE=1 to regenerate all.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

ROOT = Path(__file__).resolve().parents[1]
CERT_DIR = ROOT / "certs"

CA_KEY = CERT_DIR / "ca.key"
CA_CRT = CERT_DIR / "ca.crt"
SERVER_KEY = CERT_DIR / "server.key"
SERVER_CRT = CERT_DIR / "server.crt"
MITM_PEM = CERT_DIR / "mitm.pem"


def _new_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _key_pem(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _cert_pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _create_ca() -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = _new_key()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Varonis Local CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - dt.timedelta(minutes=1))
        .not_valid_after(_now() + dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    CA_KEY.write_bytes(_key_pem(key))
    CA_CRT.write_bytes(_cert_pem(cert))
    return key, cert


def _load_ca() -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = serialization.load_pem_private_key(CA_KEY.read_bytes(), password=None)
    cert = x509.load_pem_x509_certificate(CA_CRT.read_bytes())
    return key, cert  # type: ignore[return-value]


def _issue(
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
    common_name: str,
    sans: list[x509.GeneralName],
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = _new_key()
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - dt.timedelta(minutes=1))
        .not_valid_after(_now() + dt.timedelta(days=825))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    return key, cert


def main() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    force = os.getenv("FORCE") == "1"

    if force or not (CA_KEY.exists() and CA_CRT.exists()):
        ca_key, ca_cert = _create_ca()
        force = True  # a new CA invalidates everything it signed
        print(f"created {CA_CRT.name}, {CA_KEY.name}")
    else:
        ca_key, ca_cert = _load_ca()

    if force or not (SERVER_KEY.exists() and SERVER_CRT.exists()):
        key, cert = _issue(
            ca_key,
            ca_cert,
            "localhost",
            [
                x509.DNSName("localhost"),
                x509.DNSName("nginx"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ],
        )
        SERVER_KEY.write_bytes(_key_pem(key))
        SERVER_CRT.write_bytes(_cert_pem(cert))
        print(f"created {SERVER_CRT.name}, {SERVER_KEY.name}")

    if force or not MITM_PEM.exists():
        key, cert = _issue(ca_key, ca_cert, "mitm", [x509.DNSName("mitm")])
        MITM_PEM.write_bytes(_key_pem(key) + _cert_pem(cert))
        print(f"created {MITM_PEM.name}")

    print(f"certs ready in {CERT_DIR}")


if __name__ == "__main__":
    main()
