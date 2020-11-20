---
layout: category
title: About
category-name: business
permalink: /category/
---

<input type="checkbox" class="sidebar-checkbox" id="sidebar-checkbox">

<div class="sidebar" id="sidebar">
  <nav class="sidebar-nav">
   <h3 class="category-topic">Topics/</h3>
    {% assign sortedCategories = site.categories | sort %}
    {% for category in sortedCategories %}
     {% assign cat4url = category[0] | remove:' ' | downcase %}
     <a class="sidebar-nav-item" href="{{site.baseurl}}/category/{{cat4url}}">
        {{category[0]}}
     </a>
    {% endfor %}
  </nav>

</div>