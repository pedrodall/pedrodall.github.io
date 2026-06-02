---
layout: page
permalink: /blog/
title: blog
nav: true
nav_order: 1
_styles: >
  .post-header { display: none; }
---

<div class="post">
  <div class="header-bar">
    <h1>Blog</h1>
    <h2>Notes on GFlowNets, variational inference, and machine learning.</h2>
  </div>

  {% if site.posts.size > 0 %}
    <ul class="post-list">
      {% for post in site.posts %}
        {% if post.external_source == blank %}
          {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
        {% else %}
          {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
        {% endif %}

        <li>
          <h3>
            <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h3>
          {% if post.description %}
            <p>{{ post.description }}</p>
          {% endif %}
          <p class="post-meta">
            {{ read_time }} min read &nbsp; &middot; &nbsp; {{ post.date | date: '%B %d, %Y' }}
          </p>
          <p class="post-tags">
            <a href="{{ post.date | date: '%Y' | prepend: '/blog/' | relative_url }}">
              <i class="fa-solid fa-calendar fa-sm"></i> {{ post.date | date: '%Y' }}
            </a>
            {% if post.tags and post.tags.size > 0 %}
              &nbsp; &middot; &nbsp;
              {% for tag in post.tags %}
                <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">
                  <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}
                </a>
                {% unless forloop.last %}&nbsp;{% endunless %}
              {% endfor %}
            {% endif %}
          </p>
        </li>
      {% endfor %}
    </ul>
  {% else %}
    <p>No posts yet.</p>
  {% endif %}
</div>
