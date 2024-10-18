import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def startbot(username,password,url):
    path = "C:\\Users\\wichpol\\Desktop\\coding\\Python_personal\\chromedriver-win64\\chromedriver.exe"
    service = Service(path)
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")#https://www.automationtestinghub.com/download-chrome-driver/#google_vignette

    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("getting in login page")
    driver.get(url)
    print("Filling username")
    driver.find_element(By.ID,"login_field").send_keys(username)#github username
    print("Filling password")
    driver.find_element(By.ID,"password").send_keys(password)#github password
    print("Clicking login button")
    driver.find_element(By.NAME,"commit").click()#github login button

    print("waiting. . . .")
    time.sleep(5)#wait for 5 second
    driver.quit()

    #enter login
username = "act22971"
password = "22971act"
url = "https://github.com/login"

startbot(username,password,url)

