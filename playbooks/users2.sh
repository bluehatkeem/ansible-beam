#!/bin/bash

ansible -m user -a "name=lilhimmy state=present" alma  -b  

ansible -m authorized_key -a "user=lilhimmy state=present key=\"{{ lookup('file','/home/svc-ansible/.ssh/id_rsa.pub') }}\"" alma  -b 

ansible -m copy -a "dest=/etc/sudoers.d/lilhimmy content='lilhimmy ALL=(root) NOPASSWD : ALL'" alma -b 
