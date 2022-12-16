#!/bin/sh
sleep 20

if ! mariadb -u root -ppassword123 -h mariadb-nmo -e "USE baseball;"
then
  echo "baseball didn't exist - creating database"
  mariadb -u root -ppassword123 -h mariadb-nmo -e "CREATE DATABASE baseball;"
  mariadb -u root -ppassword123 -h mariadb-nmo -D baseball < baseball.sql
fi

echo "database exists"
echo "adding tables for final project"
#mariadb -u root -ppassword123 -h mariadb-nmo -D baseball < make_constants_table.sql
#mariadb -u root -ppassword123 -h mariadb-nmo -D baseball < final.sql

echo "generating output"
python final.py
echo "output now available"
#echo "/output/output.txt now available"