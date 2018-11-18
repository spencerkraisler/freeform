# WELCOME TO FREE FORM

# TO DO STUFF RUN FREEFORM.PY AND HOPE YOU HAVE EVERYTHING INSTALL <3 


FreeForm is an application that can turn any computer into a touchscreen tablet. I’ve always been a fan of computer vision, and one component of every computer I feel is very underestimated is the webcam. People always cover it with tape, it’s never being used to it’s maximum potential. As a result, laptop cameras have degraded over the years. However computer vision shows that regardless of how crappy your webcam is, it can still detect literally anything you want, as long as you implement the right algorithms. 

The ability for the webcam to track a stylus, or even your finger, would have huge improvements for digital commerce. Teleconferences for instance in business to business transactions. Wouldn’t it be nice for someone with an ordinary computer to just draw something on the screen on spot?
 
As of now, FreeForm takes advantage of an extremely robust yet simple computer vision algorithm called histogram backprojection. I’m also implementing some clever algorithms to make sure the cursor doesn’t go nuts. 

Ideally, I would have liked to train a single shot detector using Tensorflow, yet I was having difficulties with that so I switched to a non machine learning algorithm. 

This application also takes advantage of moving average to smooth your drawing. Funny enough, I got the math to do this from a neural network tutorial. 

Furthermore, there is a neural network working in the background. I trained one on Keras to detect certain doodles from Google’s Doodle Dataset! 


However, it seems the neural network isn’t working at the moment. I must be formatting the data wrong. 

Ideally, there are people in this world who are illiterate. I feel it’d be nice for them to draw pictures so they can buy cool stuff on amazon. Want an iphone? Draw it. Then click ‘search’ and boom, amazon pops up searching for iphones! Capitalism and computer vision go hand in hand. :D

As for scalability, this has huge potentitall. The concept of finger detection is so cool. If I had more time, I’d make this particular model from this one paper I read on finger detection. It used two neural networks to accureactly track your finger. Much more robust than a stylus.

In any case, this application is fun. It’s not some boring scheduler app, you’re drawing on the SCREEN WITH A MARKER BUT YOU’RE NOT DRAWING ON THE SCREEN! WHOA! 

