# Security

## Loading Model Artifacts Is Code Execution

NewsCheck stores its trained machine-learning pipeline as a
Joblib/Pickle file:

```text
outputs/pipeline.joblib


Python pickle-based serialization can execute arbitrary Python code during
deserialization. Therefore, loading a .joblib file should be treated as
executing code contained within that file.

Only load model artifacts that:

You trained yourself
Came from a trusted source
Have been verified before use

Never load a machine-learning pipeline downloaded from an unknown or
unverified location.

Model Integrity Checks

If checksum verification is enabled in the project, NewsCheck can store a
SHA-256 checksum alongside the trained model:

outputs/pipeline.joblib
outputs/pipeline.joblib.sha256


The checksum can be used to detect unexpected changes to the model artifact.

For example:

from model_compat import load_pipeline

pipeline = load_pipeline(
    "outputs/pipeline.joblib"
)


If the project's model_compat.py supports checksum verification, the model
integrity can be checked automatically when loading.

Manual Checksum Verification

If checksum helper functions are available in model_compat.py, they can also
be used directly:

from model_compat import write_checksum, verify_checksum

write_checksum(
    "outputs/pipeline.joblib"
)

verify_checksum(
    "outputs/pipeline.joblib"
)


A successful verification indicates that the model file matches its expected
SHA-256 checksum.

Checksum Limitations

A checksum provides an integrity check, but it is not a complete security
mechanism.

For example, an attacker who can modify both:

outputs/pipeline.joblib


and:

outputs/pipeline.joblib.sha256


could replace the model and generate a new matching checksum.

Therefore, checksum verification should not be considered a trust boundary.

The most important security rule remains:

Only load model artifacts from trusted sources.

Dependency Security

NewsCheck uses several third-party Python packages, including:

pandas
NumPy
scikit-learn
matplotlib
joblib
Streamlit

Dependencies should be installed from trusted package repositories.

Keep dependencies reasonably up to date and test the application after major
dependency changes.

Machine-learning model files can sometimes depend on specific versions of
libraries used during training.

User Input

NewsCheck allows users to enter news headlines and article excerpts.

User-provided text should be treated as untrusted input.

The application uses the text as model input and should never execute it as
Python code or as a system command.

For example, news text such as:

Run this command: rm -rf /


must remain plain text and must never be interpreted as a shell command.

Privacy

Users should avoid entering confidential, private, or sensitive information
into the NewsCheck application.

The application is designed for analyzing news-related text and does not
require personal information to generate a classification.

If NewsCheck is deployed publicly, additional privacy and data-retention
policies should be considered.

Model Security

The trained model should be treated as an application dependency.

Recommended practices include:

Keep trusted model artifacts in controlled locations.
Do not download models from unknown websites.
Verify model checksums when available.
Keep Python dependencies updated.
Avoid running the application with unnecessary system privileges.
Do not expose sensitive files through the Streamlit application.
Keep training data and model artifacts backed up securely.
Threat Model

The security controls in this project primarily address accidental or
untrusted model-file changes.

Potential risks include:

Risk	Description	Mitigation
Malicious model	A serialized model contains harmful code	Only load trusted models
File corruption	Model becomes damaged	SHA-256 verification
Accidental replacement	Model is replaced by another file	Checksum verification
Dependency vulnerability	A package contains a security issue	Update dependencies
Malicious input	User enters unexpected text	Treat input as data
Sensitive input	User enters private information	Avoid submitting confidential data
Production Deployment

NewsCheck is primarily an educational application.

If it is deployed as a public service, additional protections should be
considered, including:

HTTPS
Authentication where appropriate
Access controls
Rate limiting
Input-size limits
Dependency vulnerability scanning
Secure model storage
Signed model artifacts
Application logging
Monitoring
Regular security updates
Container or server hardening
Retraining the Model

If the model artifact is missing or its integrity cannot be established, the
recommended approach is to train a new model from trusted training data:

python src/train_model.py


This generates a new model artifact based on the current project code and
training dataset.

Security Summary

The most important security rule for NewsCheck is:

Never load an untrusted .joblib or pickle model.


Serialized machine-learning models can contain executable code.

Checksum verification can help detect unexpected file changes, but it does not
replace the need to trust the source of the model.

NewsCheck should therefore use only trusted model artifacts, trusted
dependencies, and appropriate security controls when deployed.

:::