import random, string
name_number = 10
name_length = 5

class People:
	def __init__(self,first_names, middle_names,last_names):
		self.first_names = first_names
		self.middle_names = middle_names
		self.last_names = last_names
	def print_names(self, pat):
		first_name_iter = iter(self.first_names)
		middle_name_iter = iter(self.middle_names)
		last_name_iter = iter(self.last_names)
		while True:
			try:
				if (pat==0):
					print('{} {} {}'.format(next(first_name_iter),next(middle_name_iter), next(last_name_iter)))
				elif(pat==1):
					print('{} {} {}'.format(next(last_name_iter),next(middle_name_iter), next(first_name_iter)))
				else:
					print('{}, {} {}'.format(next(last_name_iter),next(middle_name_iter), next(first_name_iter)))
					
			except:
				break	
	
	def name_pattern(self, pattern):
		if(pattern=='first_name_first'):
			self.print_names(0)
		elif(pattern=='last_name_first'):
			self.print_names(1)
		elif(pattern=='last_name_with_comma_first'):
			self.print_names(2)
		else:
			print("\n pattern not found")
	
	def sort_last_name(self):
		temp_name_list = self.last_names
		temp_name_list.sort()
		temp_name_iter = iter(temp_name_list)
		while True:
			try:
				print(next(temp_name_iter))
			except:
				break
		

class PeopleWithMoney(People):
	def __init__(self,first_names, middle_names,last_names, wealth):
		super().__init__(first_names, middle_names,last_names)
		self.wealth = wealth
	def print_names_wealth(self):
		first_name_iter = iter(self.first_names)
		middle_name_iter = iter(self.middle_names)
		last_name_iter = iter(self.last_names)
		wealth_iter = iter(self.wealth)
		while True:
			try:
				print('{} {} {} {}'.format(next(first_name_iter),next(middle_name_iter), next(last_name_iter),next(wealth_iter)))
			except:
				break	
	def sort_wealth_print(self):
		#wealth_list = self.wealth
		#wealth_list.sort()
		wealth_list = sorted(self.wealth)
		for i in range(len(wealth_list)):
			ind = self.wealth.index(wealth_list[i])
			print('{} {} {} {}'.format(self.first_names[ind],self.middle_names[ind], self.last_names[ind],wealth_list[i]))
		
		
		
#people_1 = People('Sibendu','', 'Paul')
#print(people_1.last_names)

names_list, first_names, middle_names, last_names = [], [], [], []
random.seed(0)
#generating random names
for i in range(3):#for [first_names, middle_names,last_names]
	for j in range(name_number): #to generate 10 strings for each [first_names, middle_names,last_names]
		temp_name = ''
		for k in range(name_length): #to randomly generate each letter
			#random.seed(0)
			temp_name = temp_name + random.choice(string.ascii_lowercase)
		names_list.append(temp_name)
		#print(temp_name)

first_names = names_list[0:name_number]
middle_names = names_list[name_number:name_number*2]
last_names = names_list[name_number*2:name_number*3]
name_iter = iter(names_list)
people_1 = People(first_names, middle_names,last_names)
#print(next(name_iter))
#people_1.print_names()
print("\n ===== first_name_first ===== \n")
people_1.name_pattern('first_name_first')
print("\n ===== last_name_first ===== \n")
people_1.name_pattern('last_name_first')
print("\n ===== last_name_with_comma_first ===== \n")
people_1.name_pattern('last_name_with_comma_first')
print("\n === sorted last names ===== \n")
people_1.sort_last_name()

#adding wealth
wealth = []
for i in range(name_number):
	wealth.append(random.randint(0,1000))
#print(wealth)
people_2 = PeopleWithMoney(first_names, middle_names,last_names, wealth)
print("\n ==== after adding wealth element with names ==== \n")
people_2.print_names_wealth()
print("\n ====  printing names ac to wealth ==== \n")
people_2.sort_wealth_print()

	


