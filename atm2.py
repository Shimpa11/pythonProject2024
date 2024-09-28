class ATM:
    def __init__(self):
        self.balance = 0
        self.is_authenticated = False

    def authenticate_user(self, pin):
        # In a real-world application, you would check this pin against a secure database
        # For simplicity, we'll assume the pin is always '1234'
        if pin == '1234':
            self.is_authenticated = True
            print("Authentication successful.")
        else:
            print("Authentication failed. Please try again.")

    def check_balance(self):
        if self.is_authenticated:
            print(f"Your balance is ${self.balance:.2f}")
        else:
            print("Please authenticate first.")

    def deposit(self, amount):
        if self.is_authenticated:
            if amount > 0:
                self.balance += amount
                print(f"Deposited ${amount:.2f}. New balance is ${self.balance:.2f}")
            else:
                print("Deposit amount must be positive.")
        else:
            print("Please authenticate first.")

    def withdraw(self, amount):
        if self.is_authenticated:
            if amount > 0:
                if amount <= self.balance:
                    self.balance -= amount
                    print(f"Withdrew ${amount:.2f}. New balance is ${self.balance:.2f}")
                else:
                    print("Insufficient funds.")
            else:
                print("Withdrawal amount must be positive.")
        else:
            print("Please authenticate first.")

    def exit(self):
        print("Thank you for using the ATM. Goodbye!")
        self.is_authenticated = False


def main():
    atm = ATM()

    while True:
        print("\nATM Menu")
        print("1. Authenticate")
        print("2. Check Balance")
        print("3. Deposit")
        print("4. Withdraw")
        print("5. Exit")

        choice = input("Please choose an option: ")

        if choice == '1':
            pin = input("Enter your PIN: ")
            atm.authenticate_user(pin)
        elif choice == '2':
            atm.check_balance()
        elif choice == '3':
            amount = float(input("Enter amount to deposit: "))
            atm.deposit(amount)
        elif choice == '4':
            amount = float(input("Enter amount to withdraw: "))
            atm.withdraw(amount)
        elif choice == '5':
            atm.exit()
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
