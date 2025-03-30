from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import csv
import time

# Initialize Chrome WebDriver
driver = webdriver.Chrome()

# URL of the UFC Stats website
url = "http://ufcstats.com/statistics/events/completed?page=all"

# Open the URL
driver.get(url)

# Wait for the table to load
wait = WebDriverWait(driver, 20)  # Increase timeout to 20 seconds
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody tr")))

# Find all event links and extract their URLs
event_links = driver.find_elements(By.CSS_SELECTOR, ".b-statistics__table-row a")
event_urls = [link.get_attribute('href') for link in event_links]

# Initialize set to store unique fighter names
unique_fighters = set()

# Open CSV file for writing
with open('fighter_stats.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Fighter Name', 'Record', 'Height', 'Weight', 'Reach', 'Stance', 'SLpM', 'Str. Acc', 'SApM', 'Str. Def',
                   'TD Avg','TD Acc','TD Def','Sub. Avg']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each event URL
    for event_url in event_urls[0:len(event_urls)]:
        # Open the event URL
        driver.get(event_url)
        
        # Wait for the table to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody tr")))
        
        # Find all fight rows in the table
        fight_rows = driver.find_elements(By.CSS_SELECTOR, ".b-fight-details__table-row")
        
        # Loop through each fight row
        for fight_row in fight_rows:
            # Find the fighter links
            fighter_links = fight_row.find_elements(By.CSS_SELECTOR, ".b-fight-details__table-col.l-page_align_left a")
            
            # Loop through each fighter link
            for fighter_link in fighter_links:
                fighter_name = fighter_link.text.strip()
                # Check if fighter is already scraped
                if fighter_name not in unique_fighters:
                    # Add fighter to set
                    unique_fighters.add(fighter_name)
                    
                    # Click on the fighter link
                    fighter_link.click()
                    
                    # Wait for fighter details to load
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".b-content__title-highlight")))
                    
                    # Extract fighter information
                    fighter_name_element = driver.find_element(By.CSS_SELECTOR, ".b-content__title-highlight")
                    record_element = driver.find_element(By.CSS_SELECTOR, ".b-content__title-record")
                    
                    # Check if fighter details contain expected information
                    if fighter_name_element and record_element:
                        fighter_name = fighter_name_element.text.strip()
                        record = record_element.text.strip()
                        
                        # Extract height, weight, reach, stance
                        fighter_details = driver.find_elements(By.CSS_SELECTOR, ".b-list__box-list-item")
                        height, weight, reach, stance = '', '', '', ''
                        for detail in fighter_details:
                            text = detail.text
                            if "HEIGHT" in text:
                                height = text.split(":")[1].strip()
                            elif "WEIGHT" in text:
                                weight = text.split(":")[1].strip()
                            elif "REACH" in text:
                                reach = text.split(":")[1].strip()
                            elif "STANCE" in text:
                                stance = text.split(":")[1].strip()
                        
                        # Extract SLpM, Str. Acc, SApM, Str. Def
                        striking_stats = driver.find_elements(By.CSS_SELECTOR, ".b-list__box-list-item_type_block")
                        SLpM, str_acc, SApM, str_def,TD_Avg,TD_Acc,TD_Def,Sub_Avg = '', '', '', '', '', '', '', ''
                        for stat in striking_stats:
                            text = stat.text
                            if "SLpM" in text:
                                SLpM = text.split(":")[1].strip()
                            elif "Str. Acc." in text:
                                str_acc = text.split(":")[1].strip()
                            elif "SApM" in text:
                                SApM = text.split(":")[1].strip()
                            elif "Str. Def" in text:
                                str_def = text.split(":")[1].strip()
                            elif "TD Avg" in text:
                                TD_Avg = text.split(":")[1].strip()
                            elif "TD Acc" in text:
                                TD_Acc = text.split(":")[1].strip()
                            elif "TD Def" in text:
                                TD_Def = text.split(":")[1].strip()
                            elif "Sub. Avg" in text:
                                Sub_Avg = text.split(":")[1].strip()
                        
                        # Write data to CSV
                        writer.writerow({'Fighter Name': fighter_name, 'Record': record, 'Height': height, 'Weight': weight, 'Reach': reach, 
                                         'Stance': stance, 'SLpM': SLpM, 'Str. Acc': str_acc, 'SApM': SApM, 'Str. Def': str_def,
                                          'TD Avg': TD_Avg,'TD Acc': TD_Acc,'TD Def': TD_Def,'Sub. Avg': Sub_Avg})
                    
                    # Navigate back to the event page
                    driver.back()

        # Scroll to reveal more fighters
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)  # Wait for the page to load more fighters
        
# Close the WebDriver
driver.quit()

