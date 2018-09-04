---
layout: post
title:  "Making a pet food monitoring system - Part 1"
date:   2018-09-03
excerpt: "In this tutotial series you will learn how to build a pet  food monitoring system using a Raspberry Pi, Flask REST API and OpenCV!"
image: "/images/hermes1.jpg"
---

## Motivation

So, this guy on the photo above is  my family’s dog, Hermes. Yes, he is a funny weirdo! Like many other dogs, he sometimes struggles with his food. When he gets nervous for some random reason, he can go most of the day without touching his food.

In an effort to make it easier to check if he is not eating properly, I decided to develop a system to monitor his food bowl.

Let’s get started!

## What you are going to need

* __Webcam:__ pretty much any model would do. I ’m going to be using the PS Eye because that’s what I have at hand
*  __Raspberry Pi:__ preferably a model 3, so you can easily connect it to your wi-fi network
* __Flask:__ a python micro web framework that we are going to use to build the REST API, so we can access and control  the raspberry remotely
* __OpenCV__: a popular computer vision library with python bindings

That’s it for now. If we happen to need any other libraries or frameworks in future sections I will tell you in time.

## Overview

We are going to be using a Raspberry pi 3  with a webcam to monitor the food level in the bowl with some computer vision magic! We are also going to build a server on the Raspberry pi with the Flask framework. This way, we are going to be able to make calls o the REST API and get the information about the food level (And a bunch of other useful information!) remotely.

<div class="box">
  <p>
    <b>NOTE:</b> This is just a prototype project for now.  That being the case, we are not going to be spending much time with the deployment of the platform, so the server will only be accessible locally  i.e. only devices connected to the same network.
  </p>
</div>
