#!/bin/sh
sleep 10

#echo "in bash"

if ! mariadb -u root -ppassword123 -h mariadb-nmo -e "USE baseball;"
then
  echo "baseball didn't exist - creating database"
  mariadb -u root -ppassword123 -h mariadb-nmo -e "CREATE DATABASE baseball;"
  mariadb -u root -ppassword123 -h mariadb-nmo -D baseball < baseball.sql
fi

mariadb -u root -ppassword123 -h mariadb-nmo baseball < homework6.sql
echo "generating output"
mariadb -u root -ppassword123 -h mariadb-nmo baseball -e "SELECT * FROM rolling_avg;" > /output/output.txt
echo "/output/output.txt now available"