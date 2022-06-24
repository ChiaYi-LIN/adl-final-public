#!/bin/bash
"""
Model trained with bert
"""
if [ -e ../tmp/intent_detect_bert_withBot_combined ]
then
    echo "../tmp/intent_detect_bert_withBot_combined already exist"
else
    gdown https://drive.google.com/drive/folders/1ZhGHCDqHHRBwuYU_zoPCgWx9BRNqG71L -O ../tmp/intent_detect_bert_withBot_combined --folder
fi

"""
Model trained with roberta (one sentence one data pair / sliding window)
"""
if [ -e ../tmp/intent_detect_roberta_withBot ]
then
    echo "../tmp/intent_detect_roberta_withBot already exist"
else
    gdown https://drive.google.com/drive/folders/10hf-SVXy2a6sX1XFJUZVL3VVfw1LjGkP -O ../tmp/intent_detect_roberta_withBot --folder
fi

if [ -e ../tmp/intent_detect_roberta_withBot_win3 ]
then
    echo "../tmp/intent_detect_roberta_withBot_win3 already exist"
else
    gdown https://drive.google.com/drive/folders/1WNNcZY7XsjfW9kJSMmfENxEXgpQF2H3x -O ../tmp/intent_detect_roberta_withBot_win3 --folder
fi

"""
Models trained with downsampled data
"""
if [ -e ../tmp/intent_detect_roberta_withBot_downsample ]
then
    echo "../tmp/intent_detect_roberta_withBot_downsample already exist"
else
    gdown https://drive.google.com/drive/folders/1fJQueqQN0Sf3647e-2gGHVOpPEATwDgM -O ../tmp/intent_detect_roberta_withBot_downsample --folder
fi

if [ -e ../tmp/intent_detect_roberta_withoutBot_win3_downsample ]
then
    echo "../tmp/intent_detect_roberta_withoutBot_win3_downsample already exist"
else
    gdown https://drive.google.com/drive/folders/1HI7mX-pO-Y7pCzayFWJbTPKwNAlmfNQf -O ../tmp/intent_detect_roberta_withoutBot_win3_downsample --folder
fi

if [ -e ../tmp/intent_detect_roberta_withBot_win3_downsample ]
then
    echo "../tmp/intent_detect_roberta_withBot_win3_downsample already exist"
else
    gdown https://drive.google.com/drive/folders/1MqS1PD2dJVGkejaueBdvbKIIScBei6Xp -O ../tmp/intent_detect_roberta_withBot_win3_downsample --folder
fi

