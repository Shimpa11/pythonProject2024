import time
print("Please insert your CARD")
time.sleep(1)
password=1234
balance = 5000
print("**********************************3**************************")
pin = int(input("Enter your  atm pin:"))








        # print(" Enter VALID INPUT")
print("************************************************************")
while True:

        if pin == password:

                #multiple line print
                print("""
                1==balance
                2==withdraw money
                3==deposit balance
                4==exit
                """)


                try:
                    option = int(input("Please enter your Option:"))
                except:
                    print("Please enter valid input")

                if option == 1:
                        print(f"Your current balance is:  {balance}")
                        print("************************************************************")
                if option == 2:
                        withdrawMoney = int(input("Enter  the amount:"))
                        balance = balance - withdrawMoney
                        print(f"{withdrawMoney} is debited from your account")

                        print(f"Your current balance is: {balance}")
                        print("************************************************************")

                if option == 3:
                        depositMoney = int(input("Enter  the amount:"))
                        balance = balance + depositMoney
                        print(f"{depositMoney} is credited into your account")

                        print(f"Your current balance is: {balance}")
                        print("************************************************************")

                if option == 4:
                    print('Thank you')
                    "break"
                    print("************************************************************")
                    break


        else:
                print("Please enter correct PIN")
                pin = int(input("Enter your  atm pin:"))



