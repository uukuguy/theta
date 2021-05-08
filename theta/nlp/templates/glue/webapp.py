#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import sys

import pandas as pd
import streamlit as st

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
#  from utils import retrieve_doc
#  from utils import feedback_doc
# pip install st-annotated-text
from annotated_text import annotated_text
#  from execbox import execbox

from theta.utils import SessionState

import time
random.seed(time.time())

# -------------------- UI and Options --------------------
st.write("# NER WebApp")
st.sidebar.header("Options")
top_k_reader = st.sidebar.slider("Max. number of answers",
                                 min_value=1,
                                 max_value=100,
                                 value=5,
                                 step=1)
top_k_retriever = st.sidebar.slider("Max. number of passages from retriever",
                                    min_value=1,
                                    max_value=500,
                                    value=100,
                                    step=10)
eval_mode = st.sidebar.checkbox("Evalution mode")
debug = st.sidebar.checkbox("Show debug info")


def tag_annotate_answer(answer, context):
    start_idx = context.find(answer)
    if start_idx >= 0:
        end_idx = start_idx + len(answer)
        return ''.join(context[:start_idx], (answer, "ANSWER", "#8ef"),
                       context[end_idx:])
    else:
        return context


def annotate_answer(answer, context):
    start_idx = context.find(answer)
    end_idx = start_idx + len(answer)
    annotated_text(context[:start_idx], (answer, "ANSWER", "#8ef"),
                   context[end_idx:])


# -------------------- State --------------------
state_question = SessionState.get(random_question="",
                                  random_answer='',
                                  next_question='false',
                                  run_query='false')

# -------------------- EDA --------------------
st.header("EDA")
from risklabel_data import ner_labels, prepare_samples

#  st.write("**ner_labels**", ner_labels)
label2id = {x: i + 1 for i, x in enumerate(ner_labels)}
st.write("**label2id**", label2id)

train_samples, test_samples = prepare_samples()

# -------------------- Train Samples --------------------
st.subheader("Train Samples")

train_samples.load_samples()
for sample in train_samples[100:110]:
    st.write(sample)

# -------------------- Test Samples --------------------
st.subheader("Test Samples")

test_samples.load_samples()
for sample in test_samples[100:110]:
    st.write(sample)

# -------------------- Search bar --------------------
# Search bar
#  random_question = state_question.random_question
#  question = st.text_input("Please provide your query:", value=random_question)
#  if debug:
#      question
#  state_question.random_question = question

#  if state_question and state_question.run_query:
#      run_query = state_question.run_query
#      st.button("Run")
#  else:
#      run_query = st.button("Run")
#      state_question.run_query = False
#      state_question.random_question = random_questions()[0]
#
#  if run_query:
#
#      device_standards = DeviceStandards(json_files_dir)
#      st.write("## Retrieved answers:")
#      answers = device_standards.query(question, max_answers=top_k_retriever)
#      for i, (score, json_data) in enumerate(answers):
#          passage_id = json_data['passage_id']
#          standard_num = json_data['standard_num']
#          standard_name = json_data['standard_name']
#
#          file_id = passage_id.split('-')[0]
#          clause_id = ''.join(passage_id.split('-')[1:])
#          st.write(
#              f"{i+1:02d}| 标准号：{standard_num:16} | 标准名称：{standard_name} | 条款号：{clause_id:10} | 文件号：{file_id} | score: {score:.3f}"
#          )
#
#          docid = passage_id.split('-')[0]
#          passage_id = ''.join(passage_id.split('-')[1:])
#          passage = device_standards.get_passage(docid, passage_id)
#
#          #  passage_text = passage['text']
#          #  if passage_text:
#          #      for x in question.split(' '):
#          #          if x in passage_text:
#          #              annotate_answer(x, passage_text)
#          #  passage_text
#          passage
#      #  context = "这是一次完整的流水线测试。"
#      #  context = question
#      #  answer = "流水线"
#      #  annotate_answer(answer, context)
#
#  #  execbox(body="""
#  #  """, autorun=False, line_numbers=True, key=None)
#
#  # -------------------- get passage --------------------
#  chapter_id = st.text_input("Please provide docid-passage_id:", value="")
#  if debug:
#      chapter_id
#  if chapter_id:
#      docid = chapter_id.split('-')[0]
#      passage_id = ''.join(chapter_id.split('-')[1:])
#      passage = device_standards.get_passage(docid, passage_id)
#      passage
