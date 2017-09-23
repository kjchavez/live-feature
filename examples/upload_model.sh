#!/usr/bin/env bash

MODEL_NAME=exportdir
GCLOUD_SAFE_MODEL_NAME=${MODEL_NAME//-/_}


# Google Cloud Platform (GCP) project id.
PROJECT_ID="tf-livefeature"

# The name of the Cloud ML Engine model you created on GCP.
GCLOUD_MODEL="test_prediction"

# This is my bucket. You can't use it.
# Change to the name of the bucket created on Google Cloud Storage.
# Note that it will be globally unique.
GCLOUD_BUCKET="tf-livefeature-models"

# A subdirectory in that bucket, used for putting different versions
# of the model.
PROJECT_FOLDER="test"

# Extract the latest version (names are timestamps)
VERSION="$(ls -1 /tmp/${MODEL_NAME} | tail -1)"
MODEL_VERSION_ID="${GCLOUD_SAFE_MODEL_NAME}_${VERSION}"

# Push the model to Google Cloud Storage and upload a new model on Cloud ML Engine.
# If this fails for unexpected reasons, try removing the runtime-version flag.
gsutil cp -r /tmp/${MODEL_NAME}/${VERSION} gs://${GCLOUD_BUCKET}/${PROJECT_FOLDER}/${MODEL_VERSION_ID} &&
gcloud ml-engine versions create ${MODEL_VERSION_ID} --model=${GCLOUD_MODEL} --origin=gs://${GCLOUD_BUCKET}/${PROJECT_FOLDER}/${MODEL_VERSION_ID} --runtime-version=1.2 --project=${PROJECT_ID}
echo "Uploaded version:"
echo ${MODEL_VERSION_ID}
