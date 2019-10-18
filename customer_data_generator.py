import csv
import random

file = open("customer_data_2.csv","w+")

for i in range (0, 10000):
	customer_id = str(random.randrange(1000, 9999, 1))
	age = str(random.randrange(2, 100, 1))
	sex = str(random.choice([0, 1]))
	num_titles = str(random.randrange(0, 25, 1))
	product_id = str(random.randrange(100, 999, 1))
	product_price = str(round(random.uniform(1,100), 2))

    # Test 1 - totally random
    # customer_satisfaction = str(random.choice([0, 1]))

    # Test 2 - a bit of structure that makes a bit of sense
	# if int(age) > 10 and int(num_titles) > 5 and float(product_price) < 20:
	# 	customer_satisfaction = 1
	# else:
	# 	customer_satisfaction = str(random.choice([0, 1]))

    # Test 3 - more definitive data on what makes for happy user - with still a bit of randomnes:
	if int(age) > 10 and int(num_titles) > 5 and float(product_price) < 20 and int(product_id) > 250 and int(sex) < 1:
		customer_satisfaction = 1
	elif int(age) > 17 and int(num_titles) > 8 and float(product_price) < 10 and int(product_id) < 250 and int(sex) >= 0:
		customer_satisfaction = 1
	elif int(age) > 50 and int(num_titles) < 3 and float(product_price) > 10:
		customer_satisfaction = 0
	else:
		customer_satisfaction = 0

	line = str(customer_id) + "," + str(age) + "," + str(sex) + "," +  str(num_titles) + "," + str(product_id) + "," + str(product_price) + "," + str(customer_satisfaction) + "\n"

	file.write(line)

file.close()