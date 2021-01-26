import random, string
name_number = 10
name_length = 5
random.seed(0)

first_names, middle_names, last_names, wealth_list = [], [], [], []

class People:
	def __init__(self,first_names, middle_names,last_names, pattern):
		self.first_names = first_names
		self.middle_names = middle_names
		self.last_names = last_names
		self.pattern = pattern
	#to make Peopleobj callable
	def __call__(self):
		last_name_sorted = sorted(self.last_names)
		for i in range(len(last_name_sorted)):
			print(last_name_sorted[i])
	#in order to make Peopleobj iterable
	def __iter__(self):
		return Peopleiterator(self)
		

class Peopleiterator:
	def __init__(self, peopleobj):
		self.item1 = peopleobj.first_names
		self.item2 = peopleobj.middle_names
		self.item3 = peopleobj.last_names
		self.pat = peopleobj.pattern
		self.index = -1
	def __iter__(self):
		return self
	def __next__(self):
		self.index += 1
		if self.index < len(self.item1):
			if(self.pat=="first_name_first"):
				return '{} {} {}'.format(self.item1[self.index], self.item2[self.index], self.item3[self.index])
			elif(self.pat=="last_name_first"):
				return '{} {} {}'.format(self.item3[self.index], self.item2[self.index], self.item1[self.index])
			elif(self.pat=="last_name_with_comma_first"):
				return '{} {} {}'.format(self.item3[self.index]+",", self.item2[self.index], self.item1[self.index])			
		else:
			raise StopIteration
	next = __next__

	
	
class PeopleWithMoney(People):
	def __init__(self,first_names, middle_names,last_names, pattern, wealth):
		super().__init__(first_names, middle_names,last_names, pattern)
		self.wealth = wealth
	#in order to make PeopleWithMoney obj iterable
	def __iter__(self):
		return PeopleWithMoneyiterator(self)
	def __call__(self):
		wealth_list = sorted(self.wealth)
		for i in range(len(wealth_list)):
			ind = self.wealth.index(wealth_list[i])
			print('{} {} {} {}'.format(self.first_names[ind],self.middle_names[ind], self.last_names[ind],wealth_list[i]))
		


class PeopleWithMoneyiterator:
	def __init__(self, peopleobj):
		self.item1 = peopleobj.first_names
		self.item2 = peopleobj.middle_names
		self.item3 = peopleobj.last_names
		self.item4 = peopleobj.wealth
		self.pat = peopleobj.pattern
		self.index = -1
	def __iter__(self):
		return self
	def __next__(self):
		self.index += 1
		if self.index < len(self.item1):
			return '{} {} {} {}'.format(self.item1[self.index], self.item2[self.index], self.item3[self.index], self.item4[self.index])
		else:
			raise StopIteration
	next = __next__

		
		

def gen_randomstring(length):
	temp_name = ''
	for k in range(length):
		temp_name = temp_name + random.choice(string.ascii_lowercase)
	return temp_name
	
	

def main():
	for j in range(name_number): #to generate 10 strings for each [first_names, middle_names,last_names]
		first_names.append(gen_randomstring(name_length))
		middle_names.append(gen_randomstring(name_length))
		last_names.append(gen_randomstring(name_length))
	#Task 5.1
	#print "first_name middle_name last_name"
	people_1 = People(first_names, middle_names,last_names, "first_name_first")
	for item in people_1:
		print(item)
	print()
	#Task 5.2
	#print "last_name middle_name first_name"
	people_2 = People(first_names, middle_names,last_names, "last_name_first")
	for item in people_2:
		print(item)
	print()
	#Task 5.3
	#print "last_name, middle_name first_name"
	people_3 = People(first_names, middle_names,last_names, "last_name_with_comma_first")
	for item in people_3:
		print(item)
	print()
	#Task 6
	#Making People instance callbale and printing out sorted last names
	people_1()
	print()
	for i in range(name_number):
		wealth_list.append(random.randint(0,1000))
	peoplewithmoney_1 = PeopleWithMoney(first_names, middle_names,last_names, "first_name_first", wealth_list)
	#Task 7.1
	#iterating through PeopleWithMoney instance and print individual's names and wealths
	for item in peoplewithmoney_1:
		print(item)
	print()
	#Task 7.2
	#Making PeopleWithMoney instance callable to print sorted list
	peoplewithmoney_1()
	


if __name__== "__main__":
  main()
		

