echo 'Demo!'
echo '======================'


read -n 1 -s -r -p "Press any key to begin training on WDBC"
echo 'Training WDBC...'
python3 train.py --demo WDBC
read -n 1 -s -r -p "Press any key to begin testing on WDBC"
echo 'Testing WDBC...'
python3 test.py --demo WDBC

echo ""
read -n 1 -s -r -p "Press any key to begin training on Grades"
echo 'Training Grades...'
python3 train.py --demo Grades
read -n 1 -s -r -p "Press any key to begin testing on Grades"
echo 'Testing Grades...'
python3 test.py --demo Grades


