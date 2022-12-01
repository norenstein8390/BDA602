#!/bin/sh
echo "in bash"

if ! mariadb -u root -ppassword123 -h mariadb-nmo "USE baseball;"
then
  echo "in loop"
  mariadb -u root -ppassword123 -h mariadb-nmo -e "CREATE DATABASE baseball;"
  mariadb -u root -ppassword123 -h mariadb-nmo baseball < ./baseball.sql
fi

mariadb -u root -ppassword123 -h mariadb-nmo baseball < ./homework6.sql
mariadb -u root -ppassword123 -h mariadb-nmo baseball -e "SELECT * FROM rolling_avg;" > /output/output.txt