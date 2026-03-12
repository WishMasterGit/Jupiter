(function () {
  'use strict';

  // ─── Starfield ──────────────────────────────────────────────────────
  const starfield = document.getElementById('starfield');

  function createStars() {
    const layers = [
      { count: 100, cls: 'star--distant', minSize: 1, maxSize: 2, speed: 0.02 },
      { count: 50,  cls: 'star--mid',     minSize: 2, maxSize: 3, speed: 0.05 },
      { count: 20,  cls: 'star--near',    minSize: 3, maxSize: 4, speed: 0.1 },
    ];

    layers.forEach(function (layer) {
      var container = document.createElement('div');
      container.className = 'star-layer';
      container.style.cssText = 'position:fixed;inset:0;pointer-events:none;';
      container.dataset.speed = layer.speed;

      for (var i = 0; i < layer.count; i++) {
        var star = document.createElement('div');
        star.className = 'star ' + layer.cls;
        var size = layer.minSize + Math.random() * (layer.maxSize - layer.minSize);
        var dur = 8 + Math.random() * 7;
        var o1 = 0.2 + Math.random() * 0.3;
        var o2 = 0.6 + Math.random() * 0.4;

        star.style.cssText =
          'width:' + size + 'px;height:' + size + 'px;' +
          'left:' + (Math.random() * 100) + '%;' +
          'top:' + (Math.random() * 100) + '%;' +
          '--dur:' + dur + 's;' +
          '--o1:' + o1 + ';--o2:' + o2 + ';' +
          'animation-delay:' + (-Math.random() * dur) + 's;';

        container.appendChild(star);
      }
      starfield.appendChild(container);
    });
  }

  createStars();

  // ─── Parallax scroll ────────────────────────────────────────────────
  var starLayers = starfield.querySelectorAll('.star-layer');
  var scrollTicking = false;

  window.addEventListener('scroll', function () {
    if (!scrollTicking) {
      requestAnimationFrame(function () {
        var scrollY = window.pageYOffset;
        starLayers.forEach(function (layer) {
          var speed = parseFloat(layer.dataset.speed);
          layer.style.transform = 'translateY(' + (scrollY * speed) + 'px)';
        });
        scrollTicking = false;
      });
      scrollTicking = true;
    }
  });

  // ─── Shooting stars (canvas-based) ──────────────────────────────────
  var shootingCanvas = document.getElementById('shooting-canvas');
  var ctx = shootingCanvas.getContext('2d');
  var shootingStars = [];
  var lastSpawnTime = 0;
  var nextSpawnDelay = 2000 + Math.random() * 4000;

  // Color palette — warm amber/white tones matching Jupiter theme
  var starColors = [
    { r: 245, g: 158, b: 11 },   // amber
    { r: 251, g: 191, b: 36 },   // yellow-amber
    { r: 255, g: 255, b: 255 },  // white
    { r: 253, g: 224, b: 171 },  // warm white
    { r: 234, g: 88,  b: 12 },   // orange
  ];

  function resizeShootingCanvas() {
    shootingCanvas.width = window.innerWidth;
    shootingCanvas.height = window.innerHeight;
  }

  resizeShootingCanvas();
  window.addEventListener('resize', resizeShootingCanvas);

  function ShootingStar() {
    var color = starColors[Math.floor(Math.random() * starColors.length)];
    this.r = color.r;
    this.g = color.g;
    this.b = color.b;

    // Random angle between 200-250 degrees (top-right to bottom-left feel)
    var angle = (200 + Math.random() * 50) * Math.PI / 180;
    var speed = 8 + Math.random() * 12;
    this.vx = Math.cos(angle) * speed;
    this.vy = -Math.sin(angle) * speed;

    // Start from upper portion of screen, biased right
    this.x = shootingCanvas.width * (0.3 + Math.random() * 0.7);
    this.y = Math.random() * shootingCanvas.height * 0.4;

    this.life = 1.0;
    this.decay = 0.006 + Math.random() * 0.008;
    this.size = 1.5 + Math.random() * 1.5;

    // Trail: store past positions
    this.trail = [];
    this.maxTrail = 25 + Math.floor(Math.random() * 20);
  }

  ShootingStar.prototype.update = function () {
    this.trail.push({ x: this.x, y: this.y });
    if (this.trail.length > this.maxTrail) {
      this.trail.shift();
    }

    this.x += this.vx;
    this.y += this.vy;
    this.life -= this.decay;
  };

  ShootingStar.prototype.draw = function () {
    if (this.trail.length < 2) return;

    // Draw trail with gradient opacity
    for (var i = 1; i < this.trail.length; i++) {
      var t = i / this.trail.length; // 0 = tail, 1 = head
      var alpha = t * t * this.life; // quadratic fade for smooth tail
      var width = this.size * t;

      ctx.beginPath();
      ctx.moveTo(this.trail[i - 1].x, this.trail[i - 1].y);
      ctx.lineTo(this.trail[i].x, this.trail[i].y);
      ctx.strokeStyle = 'rgba(' + this.r + ',' + this.g + ',' + this.b + ',' + alpha + ')';
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // Draw bright head with glow
    var headAlpha = this.life;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(' + this.r + ',' + this.g + ',' + this.b + ',' + headAlpha + ')';
    ctx.shadowColor = 'rgba(' + this.r + ',' + this.g + ',' + this.b + ',' + (headAlpha * 0.6) + ')';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
  };

  ShootingStar.prototype.isDead = function () {
    return this.life <= 0 ||
      this.x < -100 || this.x > shootingCanvas.width + 100 ||
      this.y < -100 || this.y > shootingCanvas.height + 100;
  };

  function shootingLoop(timestamp) {
    ctx.clearRect(0, 0, shootingCanvas.width, shootingCanvas.height);

    // Spawn new shooting stars at random intervals
    if (timestamp - lastSpawnTime > nextSpawnDelay) {
      shootingStars.push(new ShootingStar());
      lastSpawnTime = timestamp;
      nextSpawnDelay = 3000 + Math.random() * 8000;
    }

    // Update and draw
    for (var i = shootingStars.length - 1; i >= 0; i--) {
      shootingStars[i].update();
      shootingStars[i].draw();
      if (shootingStars[i].isDead()) {
        shootingStars.splice(i, 1);
      }
    }

    requestAnimationFrame(shootingLoop);
  }

  requestAnimationFrame(shootingLoop);

  // ─── Mobile nav ─────────────────────────────────────────────────────
  var menuBtn = document.getElementById('mobile-menu-btn');
  var mobileMenu = document.getElementById('mobile-menu');

  menuBtn.addEventListener('click', function () {
    mobileMenu.classList.toggle('hidden');
  });

  // Close mobile menu on link click
  mobileMenu.querySelectorAll('a').forEach(function (link) {
    link.addEventListener('click', function () {
      mobileMenu.classList.add('hidden');
    });
  });

  // ─── Navbar border on scroll ────────────────────────────────────────
  var navbar = document.getElementById('navbar');

  function updateNav() {
    if (window.pageYOffset > 20) {
      navbar.classList.add('bg-zinc-950/80', 'backdrop-blur-xl', 'border-b', 'border-white/10');
    } else {
      navbar.classList.remove('bg-zinc-950/80', 'backdrop-blur-xl', 'border-b', 'border-white/10');
    }
  }

  window.addEventListener('scroll', updateNav);
  updateNav();

  // ─── Before/After comparison slider ─────────────────────────────────
  var comparison = document.getElementById('comparison');
  var beforeLayer = comparison.querySelector('.comparison-before');
  var handle = comparison.querySelector('.comparison-handle');
  var dragging = false;

  function updateSlider(clientX) {
    var rect = comparison.getBoundingClientRect();
    var x = clientX - rect.left;
    var pct = Math.max(0, Math.min(1, x / rect.width));
    beforeLayer.style.width = (pct * 100) + '%';
    handle.style.left = 'calc(' + (pct * 100) + '% - 1.5px)';

    // Update the before image width to match container width
    var beforeImg = beforeLayer.querySelector('img');
    beforeImg.style.width = rect.width + 'px';
  }

  comparison.addEventListener('mousedown', function (e) {
    e.preventDefault();
    dragging = true;
    updateSlider(e.clientX);
  });

  window.addEventListener('mousemove', function (e) {
    if (dragging) {
      updateSlider(e.clientX);
    }
  });

  window.addEventListener('mouseup', function () {
    dragging = false;
  });

  // Touch support
  comparison.addEventListener('touchstart', function (e) {
    e.preventDefault();
    dragging = true;
    updateSlider(e.touches[0].clientX);
  }, { passive: false });

  window.addEventListener('touchmove', function (e) {
    if (dragging) {
      updateSlider(e.touches[0].clientX);
    }
  });

  window.addEventListener('touchend', function () {
    dragging = false;
  });

  // Set correct before image width on load/resize
  function syncBeforeImageWidth() {
    var rect = comparison.getBoundingClientRect();
    var beforeImg = beforeLayer.querySelector('img');
    beforeImg.style.width = rect.width + 'px';
  }

  window.addEventListener('resize', syncBeforeImageWidth);
  syncBeforeImageWidth();

  // ─── Platform detection ─────────────────────────────────────────────
  function detectPlatform() {
    var ua = navigator.userAgent.toLowerCase();
    if (ua.indexOf('mac') !== -1) return 'mac';
    if (ua.indexOf('win') !== -1) return 'windows';
    if (ua.indexOf('linux') !== -1) return 'linux';
    return null;
  }

  var platform = detectPlatform();
  if (platform) {
    var card = document.querySelector('.platform-card[data-platform="' + platform + '"]');
    if (card) {
      card.classList.add('detected');
    }
  }

  // ─── Fade-in on scroll (IntersectionObserver) ───────────────────────
  var fadeEls = document.querySelectorAll('.fade-in');

  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -40px 0px',
    });

    fadeEls.forEach(function (el) {
      observer.observe(el);
    });
  } else {
    // Fallback: show everything
    fadeEls.forEach(function (el) {
      el.classList.add('visible');
    });
  }
})();
