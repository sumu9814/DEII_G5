# GitHub Stargazer Predictor
## Prerequisities: 
Docker, Docker Compose

## How to run:
git clone https://github.com/sumu9814/DEII_G5.git

cd DEII_G5/ci_cd/development_server
  - docker compose up build
  - docker compose up -d --force-recreate
  - web app will start, port 5100
cd DEII_G5/ci_cd/production_server
  - docker compose -p star-prediction up --build -d
  - web app will start, port 5100

## Web app
There are 2 main featues for prediction:
   - upload csv with a list of features for multiple repositories
   - complete a form with the same features for a single repository

If successful, both actions lead to a results page where the predicted number of stargazers is shown in a top


## How we pushed from dev server to production
cd ~/star-prediction
git add .
git commit -m "...."
git push production master

NOTE:
1. git add . --> pushes only model_files and app_files
2. Include the below message in commit for the respective deployment:
	-- [force model] >> pushes the model to prod without comparing the accuracy
	-- [force app] >> pushes the app files to prod and restarts the docker container
	-- [force all] >> does the above two
	-- If either of the above is NOT mentioned, then the accuracy is compared and deployed
