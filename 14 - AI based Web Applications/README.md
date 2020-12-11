# Session 14 - AI-based Web Applications

The goal of this assignment is to deploy the image captioning model from the [last assignment](../12%20-%20Image%20Captioning/) on an AWS EC2 instance.

The model is served as a django web application. The django code can
be found in the [caption_portal](caption_portal/) directory and the server configuration code for AWS can be found in the [server_config](server_config/) directory.

## AWS Setup

We used an Ubuntu 20.04 EC2 instance with the following configuration:

- Instance Type: t3.large
- Storage: 16 GB General Purpose SSD
- Security groups: ssh (port 22), HTTP (port 80), Custom (port 8000)

## Hosting Django

For step-by-step instructions on how to deploy a django app on a Linux server, refer to this [blog post](https://shan18.github.io/2020-12-09-deploy-django-nginx-uwsgi/).
