#!/bin/bash

# Some keys to play with:
# F88AD855-A3E8-49CA-8D93-CB1980815DCA
# 8F93BAD3-962F-4DC4-BA92-ABF45169BDE8

key=$1

if [ -z "$key" ]
then
      echo "Please provide apikey parameter!"
else
      curl -v -H "Accept: text/javascript" https://rest.coinapi.io/v1/?apikey=$key
fi