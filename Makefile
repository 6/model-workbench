TF_DIR := remote
PLAN_FILE  := tfplan

tf-init:
	terraform -chdir=$(TF_DIR) init

tf-plan:
	terraform -chdir=$(TF_DIR) plan -out=$(PLAN_FILE)
	terraform -chdir=$(TF_DIR) show -no-color $(PLAN_FILE) | less

tf-apply:
	terraform -chdir=$(TF_DIR) apply $(PLAN_FILE)

tf-destroy:
	terraform -chdir=$(TF_DIR) destroy
