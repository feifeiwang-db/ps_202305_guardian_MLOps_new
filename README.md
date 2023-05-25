# ps_202305_guardian_MLOps_new

## Workflow for Data Scientists (in ‘preprod’ DB workspace):

* Fork the ML template

* Create a “feature” branch based from “main” branch

* When code is ready, make sure to modify the “version_to_submit” in config.py file for the desired version of model to submit, and run your “train” notebook

* Create a PR, merge from your “feature” branch to “main”

* The “staging” pipeline will run unit tests, integration tests (as a SP, to regenerate a “sp_” model), and validate that the newest version of “sp_” model is having the same accuracy as the DS submitted code/model

* Once all 3 tests are passed, the “sp_” model is automatically transitioned from “None” stage to “Staging” stage

## Workflow for Production Engineers(In preprod and prod DB workspaces) 

* Once the “sp_” model is reviewed by PE, manually transition it from “staging” stage to “production”  stage (preprod workspace)

* Now PE can approve this PR merge from “feature” to “main”, and delete the “feature” branch

* Create a PR from “main” branch to “release” branch

* It will trigger 4 notebooks to run:
  * Export: export the latest production version of “sp_” model and artifacts to S3 location (preprod)
  * Import: import from S3 to prod environment, model name will be “imported_sp_XXXX” (prod)
  * Prediction_in_production: use the logged input_example to generate a prediction by using the imported model (prod)
  * Monitoring: check if the newest imported production version of model has higher accuracy than the previous version of model, when assuming the input data schema does not change. (prod)

* Approve the PR from main to release

