---
layout: page
title: diaryrack
team: "Team: Dylan Leong, Haroun Chahed, Scott Lee Chua, Shardul Sapkota, Varun Taak"
when: August 2019 - May 2020
img: /assets/img/diaryrack/thumbnail-new.png
importance: 3
---

A chatbot and web platform to facilitate in person meetings by streamlining the process of organizing group meetings. Informal meetings are organized largely through informal channels: messenger platforms. Yet, integrated utilities like calendars and reminder systems are sparse. Existing systems rely on polling techniques which can be inefficient as they require each participant’s response and even longer to schedule from all those availabilities.  

I coordinated a team of 4 to conceptualize, design, implement, and evaluate (1) a [telegram chatbot](https://github.com/sapkotashardul/NBS){:target="_blank"}

<div class="row justify-content-sm-center">
    <div class="col-sm-5 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/diaryrack/tele-1.png' | relative_url }}" alt="" title="chat"/>
    </div>
    <div class="col-sm-5 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/diaryrack/tele-2.png' | relative_url }}" alt="" title="chat"/>
    </div>
</div>
<br>

and (2) a website that helps organizers during large and recurrent meetings with easy rsvp tracking and sharing of invites with customized links and integration to google calendar. 

<div class="row justify-content-sm-center">

<iframe width="560" height="315" src="https://www.youtube.com/embed/KntXvOeecl0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>


Both projects ultimately aimed to automate meeting scheduling by finding common free times from people’s schedules. The chatbot API was built using python. The website with Angular JS, Cassandra DB, and Python.


I separately worked on the meeting automation algorithm as a constrained optimization problem for an independent research project. Check it out [here](../IntegerProgramming){:target="_blank"}.

    Tools: Python, Angular JS, Cassandra DB