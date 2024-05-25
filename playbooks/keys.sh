#!/bin/bash

ansible -m user -a 'name=keem state=present' -b -K all

ansible -m authorized_key -a "user=keem state=present key=\"{{ lookup ( 'file', '/home/svc-ansible/.ssh/id_rsa.pub' ) }}\"" -b  -K all 
