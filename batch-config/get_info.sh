#!/bin/bash

#Get IP Adress
ifconfig wlp1s0 | grep 'inet ' | awk '{print "IP: " $2}'

# Get the MAC address with "MAC:" prefix
ifconfig | grep wlp1s0 | awk -F 'HWaddr ' '{print "MAC: " $2}'

# Get the SSID with "ID:" prefix
cat /data/overlay/current/hostapd.conf | grep ssid | awk -F 'ssid=' '{print "ID: " $2}'