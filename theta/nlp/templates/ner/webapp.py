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
#      st.write("## Retrieved answers:")
#
