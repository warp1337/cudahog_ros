#!/bin/bash

echo ">>> Usage train_cudahog.sh <path to dataset base dir> <SVM_DENSE_HOME> <cudahog_home> <name>"

export DATASET_BASEDIR=$1
export SVM_DENSE_HOME=$2
export CUDAHOG_HOME=$3
export NAME=$4

export LD_LIBRARY_PATH=$CUDAHOG_HOME/lib64:$LD_LIBRARY_PATH

if [ $# -eq 4 ]
  then
    echo "All set. Let's go"
  else
    echo "Not enough arguments. Exiting ..."
    exit 1
fi

# You might want to delete previously created feature files!
rm -f /tmp/features /tmp/features_hard
rm -rf new_model

echo ">> Creating new_model directory: "
mkdir -p new_model

echo ">> Copying example configuration file to new model directory:"
cp -f example_config new_model/config

echo "------------- "
echo "----- PHASE _1"
echo "------------- "

${CUDAHOG_HOME}/bin/cudaHOGDump --config new_model/config -a $NAME -p $DATASET_BASEDIR/pos -n $DATASET_BASEDIR/neg -o /tmp/features

echo "------------- "
echo "----- PHASE _2"
echo "------------- "

${SVM_DENSE_HOME}/bin/svm_learn -t 0 /tmp/features new_model/svm_model_upper

echo "------------- "
echo "----- PHASE _3"
echo "------------- "

${CUDAHOG_HOME}/bin/cudaHOGDump --config new_model/config -a $NAME -p $DATASET_BASEDIR/pos -n $DATASET_BASEDIR/neg -f /tmp/features -o /tmp/features_hard

echo "------------- "
echo "----- PHASE _4"
echo "------------- "

${SVM_DENSE_HOME}/bin/svm_learn -t 0 /tmp/features_hard new_model/svm_model_upper
