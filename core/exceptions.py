#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : exceptions.py
#   Author      : ronen halevy 
#   Created date:  8/11/22
#   Description :
#
# ================================================================
import sys

class NoDetectionsFound(Exception):
    """Raised when no detections found"""
    pass
