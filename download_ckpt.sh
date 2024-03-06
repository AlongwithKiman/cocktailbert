#!/bin/bash

SAVE_PATH="./ckpt"
FILE_NAME="checkpoint.ckpt"

mkdir -p "$SAVE_PATH"
cd $SAVE_PATH

FILE_ID="1olPNiRHSs1qzyHxFb72cR4Sw8OX87dSW"
gdown $FILE_ID -O "$FILE_NAME"

echo "checkpoint 다운로드 완료"
