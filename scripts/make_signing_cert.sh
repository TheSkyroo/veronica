#!/bin/bash
# Create (once) a self-signed code-signing certificate for Veronica.
#
# Why: macOS keys Microphone, Screen Recording and Accessibility grants to an
# app's code signature. An ad-hoc signature is just a hash of the bundle, so
# every rebuild that changes a single byte looks like a different app and the
# permission prompts come back. Signed with a stable certificate instead, the
# grant is keyed on the bundle id plus this certificate, and it survives every
# rebuild.
#
# Run once:   ./scripts/make_signing_cert.sh
# Then:       make app      (build_app picks the identity up automatically)
#
# macOS will ask for your login password when the certificate is marked as
# trusted for code signing — that is the only interactive step.
set -euo pipefail

NAME="${VERONICA_SIGN_IDENTITY:-Veronica Local Signing}"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-identity -v -p codesigning | grep -qF "$NAME"; then
    echo "make_signing_cert: '$NAME' already exists — nothing to do."
    exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/openssl.cnf" <<EOF
[ req ]
distinguished_name = dn
x509_extensions    = v3
prompt             = no

[ dn ]
CN = $NAME

[ v3 ]
basicConstraints       = critical,CA:false
keyUsage               = critical,digitalSignature
extendedKeyUsage       = critical,codeSigning
subjectKeyIdentifier   = hash
EOF

# 10 years: this never leaves the Mac, and an expiry would silently start the
# permission prompts again.
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout "$work/key.pem" -out "$work/cert.pem" -config "$work/openssl.cnf" 2>/dev/null
# -legacy: OpenSSL 3 defaults to AES-256-CBC + PBKDF2, which Security.framework
# cannot read ("MAC verification failed during PKCS12 import"); the older
# RC2/3DES encoding is what `security import` understands. A pass phrase is
# used because an empty one trips the same check.
P12_PASS="veronica"
openssl pkcs12 -export -legacy -inkey "$work/key.pem" -in "$work/cert.pem" \
    -name "$NAME" -out "$work/cert.p12" -passout "pass:$P12_PASS" 2>/dev/null

# -T /usr/bin/codesign: codesign may use the key without asking every time.
security import "$work/cert.p12" -k "$KEYCHAIN" -P "$P12_PASS" -T /usr/bin/codesign -A >/dev/null
echo "make_signing_cert: imported '$NAME' into the login keychain."

echo "make_signing_cert: marking it trusted for code signing (macOS will ask for your login password)…"
security add-trusted-cert -r trustRoot -p codeSign -k "$KEYCHAIN" "$work/cert.pem"

if security find-identity -v -p codesigning | grep -qF "$NAME"; then
    echo "make_signing_cert: done — '$NAME' is ready. Run 'make app' and grant"
    echo "                   Microphone and Screen Recording one last time."
else
    echo "make_signing_cert: the certificate was imported but is not yet a valid"
    echo "                   code-signing identity. Open Keychain Access, find"
    echo "                   '$NAME', and set 'Code Signing' to 'Always Trust'." >&2
    exit 1
fi
