---
layout: post
title:  "Making a pet food monitoring system - Part 1"
date:   2018-09-03
excerpt: "In this tutotial series you will learn how to build a pet  food monitoring system using a Raspberry Pi, Flask REST API and OpenCV!"
image: "/images/hermes1.jpg"
---

## Motivation

So, this guy on the photo above is  my family’s dog, Hermes. Yes, he is a funny weirdo! Like many other dogs, he sometimes struggles with his food. When he gets nervous, for some random reason, he can go most of the day without touching his food.

In an effort to make it easier to check if he is not eating properly, I decided to develop a system to monitor his food bowl.

Let’s get started!

## What you are going to need

* __Webcam:__ pretty much any model would do. I ’m going to be using the PS Eye because that’s what I have at hand
*  __Raspberry Pi:__ preferably a model 3, so you can easily connect it to your wi-fi network
* __Flask:__ a python micro web framework that we are going to use to build the REST API, so we can access and control  the raspberry remotely
* __OpenCV__: a popular computer vision library with python bindings

That’s it for now. If we happen to need any other libraries or frameworks in future sections I will let you know.

## Overview

We are going to be using a Raspberry pi 3  with a webcam to monitor the food level in the bowl with some computer vision magic! We are also going to build a server on the Raspberry pi with the Flask framework. This way, we are going to be able to make calls to the REST API and get the information about the food level (And a bunch of other useful information!) remotely.

<div class="box">
  <p>
    <b>NOTE:</b> This is just a prototype project for now.  That being the case, we are not going to be spending much time with the deployment of the platform, so the server will only be accessible locally  i.e. only devices connected to the same network will be able to make requests.
  </p>
</div>

## Installation process

I will not attempt to reinvent the wheel, instead of presenting detailed explanation on how to install every dependency, I will provide some guidance and links to useful tutorials that are way more well explained than I could ever make.

__1. Installing Raspbian on the Raspberry Pi:__

This is the first step that you will have to make, and it’s pretty simple. You will need a formatted micro SD card (8 gb or more) to install the Raspbian operating system.

Raspbian is a Linux distribution especially made for the Raspberry Pi and they provide a very useful tool for installing it called NOOBS. The step by step guide for installing the OS though this tool can be found on the [Raspberry Pi website](https://projects.raspberrypi.org/en/projects/noobs-install).

__2. Installing the Flask Micro-framework:__

The Raspbian OS comes with Python pre-installed, so you won’t need to worry about that! The next step is to install the Flask framework. Luckly for us, it’s a pretty simple process, you only need to make one command on the terminal:

```
~$ pip install -U Flask
```

That’s it! For more information please visit the [project webpage](http://flask.pocoo.org/) where you can find example projects and a quick start tutorial.

__3. Installing OpenCV:__

Well, I'm not going to lie to you: installing OpenCV sucks! This is because of the multiple dependencies and pre requisites that have to be installed for the library to work correctly.

That being the case, I highly recommend that you check out Adrian Rosebrock’s truly AWESOME tutorial on how to install the library on the latest version of Raspbian. The tutorial is available on his site, Pyimagesearch, on this [link](https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/).

## Next steps

In the next part of this tutorial we will learn how to make a boilerplate Flask application and start our REST API. After that we will develop the python script to detect the food on the bowl using OpenCV and integrate it on our API. 
