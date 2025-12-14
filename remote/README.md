Setup

```sh
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
```

Then, fill out .envrc in root dir. Finally run:

```
make tf-init
make tf-plan
make tf-apply
```
