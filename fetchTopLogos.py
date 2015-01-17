import urllib, urllib2, json

def main():
	# Fetch and parse JSON from the Wikia API
	print("Fetching JSON...")
	raw = urllib2.urlopen("http://logos.wikia.com/api/v1/Articles/Top/?limit=250").read()
	print("Converting into JSON...")
	jsonified = json.loads(raw)

	# Scrape each page found for image URLs and download them
	print("Iterating over list of pages...")
	errorCount = 0
	successCount = 0
	erroredSites = []
	for site in jsonified['items']:
		count = 1
		print("Looking at page: " + site['url'])
		lines = urllib2.urlopen("http://logos.wikia.com" + site['url']).readlines()
		foundImage = False
		url = ""
		for line in lines:
			if "-present" in line and "mw-headline" in line:
				foundImage = True

			if foundImage and "<img" in line:
				#print(" -> " + line) # for debugging
				try:
					url = line.split("data-src=\"")[1].split("\"")[0]
					print(" -> Downloading logo...")
					urllib.urlretrieve(url, "./logos/" + site['url'].split("/wiki/")[1].split("/")[0] + str(count))
					count += 1
					successCount += 1
					print(" -> Download complete!")
				except:
					errorCount += 1
					erroredSites.append(site['url'])

				print("Successes to date: " + str(successCount))
				print("Errors to date: " + str(errorCount))

				foundImage = False

	print("---[ Process Complete! ]---")
	print(" Errors: " + str(errorCount))
	print(" Successes: " + str(successCount))
	print(" Errored pages: " + erroredSites)
	print()

main()