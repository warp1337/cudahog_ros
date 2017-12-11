#!/bin/zsh

echo "change the INRIA variable to point to the INRIAPerson dataset."
echo "make sure that the SVMdense package is present on your system."

export INRIA=/work/images/INRIAPerson/train_64x128_H96/
SVMDENSE=${HOME}/work/svmdense
CUDAHOG=../

# you might want to delete previously created feature files!
#rm -f /tmp/features /tmp/features_hard
#rm -rf mymodel
echo "creating mymodel directory: "
mkdir mymodel

echo "copying working configuration file to new model directory:"
cp ${CUDAHOG}/model/config mymodel/

${CUDAHOG}/bin/cudaHOGDump --config mymodel/config -a pedestrian \
	-p $INRIA/pos -n $INRIA/neg -o /tmp/features

${SVMDENSE}/svm_learn -t 0 /tmp/features mymodel/svm_model

${CUDAHOG}/bin/cudaHOGDump --config mymodel/config -a pedestrian \
	-p $INRIA/pos -n $INRIA/neg -f /tmp/features -o /tmp/features_hard

${SVMDENSE}/svm_learn -t 0 /tmp/features_hard mymodel/svm_model


