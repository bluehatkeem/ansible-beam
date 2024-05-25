#!/bin/bash

export HOST=$(hostname | cut -d. -f1)

hostnamectl set-hostname $HOST.kubeworld.io
