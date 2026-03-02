(function () {
  const CARD_PLACEHOLDER = "/static/cover-fallback.svg";
  const PLAYER_PLACEHOLDER = "/static/cover-fallback.svg";
  const DEFAULT_ACCOUNT_AVATAR = "https://i.pravatar.cc/300?img=32";
  const USER_ID_KEY = "eraex_user_id";
  const PLAYLISTS_SELECTED_KEY_PREFIX = "eraex_playlist_selected_v1";
  const DEFAULT_PLAYLIST_ID = "pl_default";
  const DEFAULT_PLAYLIST_NAME = "My Playlist";
  const REPEAT_ONE_STORAGE_KEY = "eraex_repeat_one";
  const SHUFFLE_STORAGE_KEY = "eraex_shuffle_queue";
  const SEARCH_PAGE_SIZE = 15;
  const HOME_QUEUE_BATCH_SIZE = 15;
  const HOME_QUEUE_MAX_TRACKS = 120;
  const HOME_QUEUE_AUTOFETCH_MAX_PAGES = 3;
  const HOME_QUEUE_SEARCH_TIMEOUT_MS = 30000;
  const FOR_YOU_VISIBLE_LIMIT = 48;
  const FOR_YOU_TARGET_ROWS = 5;
  const FOR_YOU_FETCH_POOL_SIZE = 160;
  const FOR_YOU_PAGE_SIZE = 15;
  const FOR_YOU_ROWS_PER_PAGE = 4;
  const PROFILE_LIBRARY_FETCH_SIZE = 120;
  const FOR_YOU_SEEN_IDS_KEY = "eraex_foryou_seen_ids";
  const FOR_YOU_SEEN_IDS_MAX = 1200;
  const LYRICS_PREFETCH_CONCURRENCY = 1;
  const LYRICS_PREFETCH_TRACK_LIMIT_FOR_YOU = 0;
  const LYRICS_PREFETCH_TRACK_LIMIT_SEARCH = 0;
  const LYRICS_PREFETCH_MARK_TTL_MS = 5 * 60 * 1000;
  const LYRICS_PREFETCH_DEFER_MS = 1800;
  const LYRICS_PREFETCH_TIMEOUT_MS = 4500;
  const FOR_YOU_PRERESOLVE_MAX_WEAK = FOR_YOU_VISIBLE_LIMIT;
  const FOR_YOU_PRERESOLVE_TIMEOUT_MS = 3200;
  const YT_EMBED_BLOCK_TTL_MS = 30 * 60 * 1000;
  const YT_EMBED_BLOCK_MAX_ATTEMPTS_PER_TRACK = 4;
  const SPOTIFY_STATUS_POLL_MS = 2000;
  // If a row already provides a video_id, stick to that exact upload (no alternate fallback search).
  const STRICT_METADATA_VIDEO_ID_PLAYBACK = true;
  const MOOD_TERMS = [
    "sad",
    "happy",
    "chill",
    "vibe",
    "energy",
    "workout",
    "study",
    "relax",
    "lofi",
    "lo-fi",
    "midnight",
    "nostalgic",
    "melancholy",
    "upbeat",
    "calm",
    "focus",
    "romantic",
    "dark",
    "dreamy",
  ];

  const $ = {
    navItems: Array.from(document.querySelectorAll(".nav-item[data-view]")),
    viewPanels: Array.from(document.querySelectorAll(".app-view[data-view-panel]")),
    query: document.getElementById("query"),
    searchBtn: document.getElementById("search-btn"),
    homeQueryForm: document.getElementById("home-query-form"),
    homeQuery: document.getElementById("home-query"),
    homeQuerySubmit: document.getElementById("home-query-submit"),
    homeStatus: document.getElementById("home-status"),
    homeQueueCaption: document.getElementById("home-queue-caption"),
    homeQueueResults: document.getElementById("home-queue-results"),
    homeQueuePager: document.getElementById("home-queue-pager"),
    homeQueuePrevBtn: document.getElementById("home-queue-prev-btn"),
    homeQueueNextBtn: document.getElementById("home-queue-next-btn"),
    homeQueuePageLabel: document.getElementById("home-queue-page-label"),
    homeQueueClearBtn: document.getElementById("home-queue-clear-btn"),
    homeChips: Array.from(document.querySelectorAll(".landing-chip[data-home-query]")),
    homeView: document.getElementById("view-home"),
    chips: Array.from(document.querySelectorAll(".chip")),
    status: document.getElementById("status"),
    results: document.getElementById("results"),
    searchPager: document.getElementById("search-pager"),
    searchPrevBtn: document.getElementById("search-prev-btn"),
    searchNextBtn: document.getElementById("search-next-btn"),
    searchPageLabel: document.getElementById("search-page-label"),
    forYouGrid: document.getElementById("for-you-grid"),
    forYouPager: document.getElementById("for-you-pager"),
    forYouPrevBtn: document.getElementById("for-you-prev-btn"),
    forYouNextBtn: document.getElementById("for-you-next-btn"),
    forYouPageLabel: document.getElementById("for-you-page-label"),
    forYouStatus: document.getElementById("for-you-status"),
    forYouSummary: document.getElementById("for-you-summary"),
    refreshFeedBtn: document.getElementById("refresh-feed-btn"),
    likedStatus: document.getElementById("liked-status"),
    likedResults: document.getElementById("liked-results"),
    historyStatus: document.getElementById("history-status"),
    historyResults: document.getElementById("history-results"),
    playlistsStatus: document.getElementById("playlists-status"),
    playlistsSummary: document.getElementById("playlists-summary"),
    playlistsResults: document.getElementById("playlists-results"),
    playlistTabs: document.getElementById("playlist-tabs"),
    playlistCoverInput: document.getElementById("playlist-cover-input"),
    playlistNewBtn: document.getElementById("playlist-new-btn"),
    playlistRenameBtn: document.getElementById("playlist-rename-btn"),
    playlistEditCoverBtn: document.getElementById("playlist-edit-cover-btn"),
    playlistClearBtn: document.getElementById("playlist-clear-btn"),
    playlistDeleteBtn: document.getElementById("playlist-delete-btn"),
    playlistModal: document.getElementById("playlist-picker-modal"),
    playlistModalClose: document.getElementById("playlist-picker-close"),
    playlistModalTitle: document.getElementById("playlist-picker-title"),
    playlistModalSubtitle: document.getElementById("playlist-picker-subtitle"),
    playlistModalStatus: document.getElementById("playlist-picker-status"),
    playlistModalName: document.getElementById("playlist-picker-name"),
    playlistModalCreateBtn: document.getElementById("playlist-picker-create-btn"),
    playlistModalList: document.getElementById("playlist-picker-list"),
    userPill: document.getElementById("user-pill"),
    player: document.getElementById("player"),
    playerArt: document.getElementById("player-art"),
    playerTitle: document.getElementById("player-title"),
    playerArtist: document.getElementById("player-artist"),
    playerDescription: document.getElementById("player-description"),
    rightPlayerArt: document.getElementById("right-player-art"),
    rightPlayerTitle: document.getElementById("right-player-title"),
    rightPlayerArtist: document.getElementById("right-player-artist"),
    playerPlaylistBtn: document.getElementById("player-playlist-btn"),
    likeBtn: document.getElementById("like-btn"),
    lyricsScroll: document.querySelector(".lyrics-scroll-container"),
    lyricsContent: document.getElementById("lyrics-content"),
    prevBtn: document.getElementById("prev-btn"),
    shuffleBtn: document.getElementById("shuffle-btn"),
    playBtn: document.getElementById("play-btn"),
    nextBtn: document.getElementById("next-btn"),
    repeatBtn: document.getElementById("repeat-btn"),
    seek: document.getElementById("seek"),
    volume: document.getElementById("volume"),
    timeCurrent: document.getElementById("time-current"),
    timeTotal: document.getElementById("time-total"),
    userAvatars: Array.from(document.querySelectorAll(".user-avatar-img")),
    authModal: document.getElementById("auth-modal"),
    authModalClose: document.getElementById("auth-modal-close"),
    authModalStatus: document.getElementById("auth-modal-status"),
    authModalSubtitle: document.getElementById("auth-modal-subtitle"),
    authModalForm: document.getElementById("auth-form"),
    authSessionView: document.getElementById("auth-session-view"),
    authProfileBanner: document.getElementById("auth-profile-banner"),
    authProfileAvatar: document.getElementById("auth-profile-avatar"),
    authChangeBannerBtn: document.getElementById("auth-change-banner-btn"),
    authChangeAvatarBtn: document.getElementById("auth-change-avatar-btn"),
    authBannerInput: document.getElementById("auth-banner-input"),
    authAvatarInput: document.getElementById("auth-avatar-input"),
    authSessionUsername: document.getElementById("auth-session-username"),
    authSessionUserId: document.getElementById("auth-session-userid"),
    authUsername: document.getElementById("auth-username"),
    authPassword: document.getElementById("auth-password"),
    authLoginBtn: document.getElementById("auth-login-btn"),
    authRegisterBtn: document.getElementById("auth-register-btn"),
    authLogoutBtn: document.getElementById("auth-logout-btn"),
    confirmModal: document.getElementById("confirm-modal"),
    confirmModalClose: document.getElementById("confirm-modal-close"),
    confirmModalTitle: document.getElementById("confirm-modal-title"),
    confirmModalMessage: document.getElementById("confirm-modal-message"),
    confirmModalCancelBtn: document.getElementById("confirm-modal-cancel-btn"),
    confirmModalConfirmBtn: document.getElementById("confirm-modal-confirm-btn"),
    settingsMenu: document.getElementById("settings-menu"),
    settingsSupportLink: document.getElementById("settings-support-link"),
    settingsButtons: Array.from(
      document.querySelectorAll('.icon-btn[aria-label=\"Settings\"]'),
    ),
  };

  const state = {
    yt: null,
    lists: {
      home: [],
      search: [],
      foryou: [],
      liked: [],
      history: [],
      playlists: [],
    },
    activeListKey: "home",
    activeView: "home",
    queueIndex: -1,
    current: null,
    lyricsData: null,
    lyricsLineEls: null,
    lyricsMode: "none", // "none" | "plain" | "synced"
    activeLyricIndex: -1,
    searching: null,
    queryId: 0,
    isPlaying: false,
    shuffleEnabled: false,
    shufflePools: new Map(),
    repeatOne: false,
    isSeeking: false,
    activePlaybackProvider: "youtube",
    fallbackTried: false,
    fallbackInFlight: false,
    ytBlockedVideoIds: new Map(),
    progressTimer: null,
    pendingStart: false,
    prefetchTimer: null,
    feedRefreshTimer: null,
    pendingPlayRecordSongId: "",
    searchCache: new Map(),
    enrichCache: new Map(),
    coverResolveCache: new Map(),
    coverResolvePending: new Map(),
    lyricsPrefetchQueue: [],
    lyricsPrefetchPending: new Set(),
    lyricsPrefetchDoneAt: new Map(),
    lyricsPrefetchActive: 0,
    lyricsPrefetchTimer: null,
    lyricsRequestId: 0,
    userId: "",
    profile: {},
    likedSongIds: new Set(),
    dislikedSongIds: new Set(),
    pendingLikeSongIds: new Set(),
    pendingDislikeSongIds: new Set(),
    playlistsLoaded: false,
    playlists: {
      selectedId: DEFAULT_PLAYLIST_ID,
      items: [],
    },
    playlistPicker: {
      open: false,
      track: null,
      busy: false,
      mode: "add", // "add" | "rename"
      targetPlaylistId: "",
    },
    playlistCoverUpload: {
      targetPlaylistId: "",
      busy: false,
    },
    searchPaging: {
      query: "",
      endpoint: "",
      pageSize: SEARCH_PAGE_SIZE,
      currentPage: 0,
      pages: new Map(),
      pending: new Map(),
      lastPageIndex: null,
      activeQueryId: 0,
      isPageLoading: false,
    },
    homePaging: {
      pageSize: HOME_QUEUE_BATCH_SIZE,
      currentPage: 0,
    },
    forYouPaging: {
      pageSize: FOR_YOU_PAGE_SIZE,
      currentPage: 0,
      tracks: [],
      loading: false,
      exhausted: false,
    },
    forYouDeepRefreshTimer: null,
    forYouDeepRefreshLastAt: 0,
    settingsMenu: {
      open: false,
      anchorButton: null,
    },
    auth: {
      statusKnown: false,
      authenticated: false,
      username: "",
      userId: "",
      bannerImage: "",
      avatarImage: "",
      busy: false,
    },
    confirmDialog: {
      open: false,
      resolver: null,
    },
    homeQueueSource: {
      query: "",
      endpoint: "",
      nextOffset: 0,
      exhausted: true,
      loading: false,
      pendingTracks: [],
    },
    spotify: {
      statusKnown: false,
      enabled: false,
      authenticated: false,
      sdkScriptLoaded: false,
      sdkLoadPromise: null,
      player: null,
      playerReady: false,
      playerConnecting: false,
      playerReadyPromise: null,
      playerDeviceId: "",
      token: "",
      tokenExpiresAt: 0,
      playbackState: null,
      progressSnapshot: null,
      authPopup: null,
      authPollTimer: null,
      fallbackInFlight: false,
      pendingAuthRetryTrackKey: "",
      pendingAuthRetryReasonCode: null,
    },
  };

  function setStatus(text, isError) {
    if (!$.status) return;
    $.status.textContent = text || "";
    $.status.classList.toggle("error", Boolean(isError));
  }

  function setForYouStatus(text, isError) {
    if (!$.forYouStatus) return;
    $.forYouStatus.textContent = text || "";
    $.forYouStatus.classList.toggle("error", Boolean(isError));
  }

  function setLikedStatus(text, isError) {
    if (!$.likedStatus) return;
    $.likedStatus.textContent = text || "";
    $.likedStatus.classList.toggle("error", Boolean(isError));
  }

  function setHistoryStatus(text, isError) {
    if (!$.historyStatus) return;
    $.historyStatus.textContent = text || "";
    $.historyStatus.classList.toggle("error", Boolean(isError));
  }

  function setPlaylistsStatus(text, isError) {
    if (!$.playlistsStatus) return;
    $.playlistsStatus.textContent = text || "";
    $.playlistsStatus.classList.toggle("error", Boolean(isError));
  }

  function setPlaylistModalStatus(text, isError) {
    if (!$.playlistModalStatus) return;
    $.playlistModalStatus.textContent = text || "";
    $.playlistModalStatus.classList.toggle("error", Boolean(isError));
  }

  function setHomeStatus(text, isError) {
    if (!$.homeStatus) return;
    $.homeStatus.textContent = text || "";
    $.homeStatus.classList.toggle("error", Boolean(isError));
  }

  function setAuthModalStatus(text, isError) {
    if (!$.authModalStatus) return;
    $.authModalStatus.textContent = text || "";
    $.authModalStatus.classList.toggle("error", Boolean(isError));
  }

  function setAuthBusy(isBusy) {
    const busy = Boolean(isBusy);
    state.auth.busy = busy;
    if ($.authLoginBtn) $.authLoginBtn.disabled = busy;
    if ($.authRegisterBtn) $.authRegisterBtn.disabled = busy;
    if ($.authLogoutBtn) $.authLogoutBtn.disabled = busy;
    if ($.authChangeBannerBtn) $.authChangeBannerBtn.disabled = busy;
    if ($.authChangeAvatarBtn) $.authChangeAvatarBtn.disabled = busy;
  }

  function friendlyAuthApiError(error, fallbackText) {
    const fallback = String(fallbackText || "Authentication request failed.");
    const status = Number(error && error.status ? error.status : 0);
    const raw = String((error && error.message) || "").trim();
    const looksHtml =
      Boolean(error && error.isHtml) ||
      /^<!doctype|^<html/i.test(raw);

    if (status === 404 || looksHtml) {
      return "Auth API not found (404). Restart the Flask backend and try again.";
    }
    if (status === 401) {
      return "Invalid username or password.";
    }
    if (status === 409) {
      return raw || "Username already exists.";
    }
    if (!raw) return fallback;
    return raw.length > 180 ? `${raw.slice(0, 177)}...` : raw;
  }

  function activeIdentityLabel() {
    if (state.auth.authenticated && state.auth.username) {
      return state.auth.username;
    }
    if (state.userId) return `User ${state.userId.slice(0, 8)}`;
    return "Guest";
  }

  function accountAvatarSrc() {
    const candidate = String(state.auth.avatarImage || "").trim();
    return candidate || DEFAULT_ACCOUNT_AVATAR;
  }

  function accountBannerSrc() {
    return String(state.auth.bannerImage || "").trim();
  }

  function applyAuthProfileMedia() {
    if ($.authProfileAvatar) {
      const avatarSrc = accountAvatarSrc();
      if ($.authProfileAvatar.src !== avatarSrc) {
        $.authProfileAvatar.src = avatarSrc;
      }
    }
    if ($.authProfileBanner) {
      const bannerSrc = accountBannerSrc();
      if (bannerSrc) {
        if ($.authProfileBanner.src !== bannerSrc) {
          $.authProfileBanner.src = bannerSrc;
        }
        $.authProfileBanner.hidden = false;
      } else {
        $.authProfileBanner.hidden = true;
      }
    }
  }

  function applyUserAvatarImages() {
    const avatarSrc = accountAvatarSrc();
    $.userAvatars.forEach(function (avatar) {
      if (!avatar) return;
      if (avatar.src !== avatarSrc) {
        avatar.src = avatarSrc;
      }
    });
    applyAuthProfileMedia();
  }

  function updateIdentityUI() {
    if ($.userPill) $.userPill.textContent = activeIdentityLabel();
    $.userAvatars.forEach(function (avatar) {
      if (!avatar) return;
      avatar.classList.toggle("is-authenticated", Boolean(state.auth.authenticated));
      avatar.title = state.auth.authenticated
        ? `${state.auth.username || "Account"} (click for account options)`
        : "Login / Create account";
    });
    applyUserAvatarImages();
  }

  function renderAuthModal() {
    const authenticated = Boolean(state.auth.authenticated && state.auth.userId);
    if ($.authSessionView) $.authSessionView.hidden = !authenticated;
    if ($.authModalForm) $.authModalForm.hidden = authenticated;
    if ($.authModalSubtitle) {
      $.authModalSubtitle.textContent = authenticated
        ? "You are signed in. Logout to switch account."
        : "Login or create an account to keep your profile data permanent.";
    }
    if ($.authSessionUsername) {
      $.authSessionUsername.textContent = String(state.auth.username || "User");
    }
    if ($.authSessionUserId) {
      $.authSessionUserId.textContent = authenticated
        ? `ID ${String(state.auth.userId).slice(0, 8)}`
        : "";
    }
    applyAuthProfileMedia();
    if (!authenticated) {
      if ($.authSessionUsername) $.authSessionUsername.textContent = "";
    }
  }

  function openAuthModal() {
    if (!$.authModal) return;
    void refreshAuthSession(true);
    renderAuthModal();
    setAuthModalStatus("");
    $.authModal.classList.add("open");
    $.authModal.setAttribute("aria-hidden", "false");
    if (!state.auth.authenticated && $.authUsername) {
      setTimeout(function () {
        try {
          $.authUsername.focus();
        } catch (error) {}
      }, 0);
    }
  }

  function closeAuthModal() {
    if (!$.authModal) return;
    $.authModal.classList.remove("open");
    $.authModal.setAttribute("aria-hidden", "true");
    setAuthModalStatus("");
  }

  function closeConfirmDialog(confirmed) {
    const resolver = state.confirmDialog && state.confirmDialog.resolver;
    state.confirmDialog.open = false;
    state.confirmDialog.resolver = null;
    if ($.confirmModal) {
      $.confirmModal.classList.remove("open");
      $.confirmModal.setAttribute("aria-hidden", "true");
    }
    if (typeof resolver === "function") {
      resolver(Boolean(confirmed));
    }
  }

  function openConfirmDialog(options) {
    const opts = options || {};
    const title = String(opts.title || "Confirm Action").trim() || "Confirm Action";
    const message = String(opts.message || "Are you sure you want to continue?").trim();
    const confirmText = String(opts.confirmText || "Confirm").trim() || "Confirm";
    const cancelText = String(opts.cancelText || "Cancel").trim() || "Cancel";
    const tone = String(opts.tone || "default").trim().toLowerCase();

    if (
      !$.confirmModal ||
      !$.confirmModalTitle ||
      !$.confirmModalMessage ||
      !$.confirmModalConfirmBtn
    ) {
      return Promise.resolve(window.confirm(message || title));
    }

    if (state.confirmDialog.open && typeof state.confirmDialog.resolver === "function") {
      state.confirmDialog.resolver(false);
      state.confirmDialog.resolver = null;
    }

    $.confirmModalTitle.textContent = title;
    $.confirmModalMessage.textContent = message;
    if ($.confirmModalCancelBtn) $.confirmModalCancelBtn.textContent = cancelText;
    $.confirmModalConfirmBtn.textContent = confirmText;
    $.confirmModalConfirmBtn.classList.toggle("is-danger", tone === "danger");
    $.confirmModalConfirmBtn.classList.toggle("is-warning", tone === "warning");

    state.confirmDialog.open = true;
    $.confirmModal.classList.add("open");
    $.confirmModal.setAttribute("aria-hidden", "false");

    setTimeout(function () {
      try {
        $.confirmModalConfirmBtn.focus();
      } catch (error) {}
    }, 0);

    return new Promise(function (resolve) {
      state.confirmDialog.resolver = resolve;
    });
  }

  function authFormCredentials() {
    const username = String($.authUsername && $.authUsername.value ? $.authUsername.value : "").trim();
    const password = String($.authPassword && $.authPassword.value ? $.authPassword.value : "");
    return { username, password };
  }

  async function refreshAuthSession(isSilent) {
    try {
      const payload = await requestJSON(
        "/api/auth/session",
        {},
        { timeoutMs: 5000, retries: 0 },
      );
      const user = payload && payload.user ? payload.user : {};
      const authenticated = Boolean(payload && payload.authenticated && user.user_id);
      state.auth.statusKnown = true;
      state.auth.authenticated = authenticated;
      state.auth.username = authenticated ? String(user.username || "") : "";
      state.auth.userId = authenticated ? String(user.user_id || "") : "";
      state.auth.bannerImage = authenticated ? String(user.banner_image || "") : "";
      state.auth.avatarImage = authenticated ? String(user.avatar_image || "") : "";
      if (authenticated) {
        setAuthModalStatus("");
      }
      updateIdentityUI();
      updatePersonalSummary();
      renderAuthModal();
      return state.auth;
    } catch (error) {
      state.auth.statusKnown = false;
      state.auth.authenticated = false;
      state.auth.username = "";
      state.auth.userId = "";
      state.auth.bannerImage = "";
      state.auth.avatarImage = "";
      updateIdentityUI();
      updatePersonalSummary();
      renderAuthModal();
      if (!isSilent) {
        setAuthModalStatus("Could not verify session right now.", true);
      }
      return state.auth;
    }
  }

  function resetUserScopedStateForIdentity() {
    if (state.forYouDeepRefreshTimer) {
      clearTimeout(state.forYouDeepRefreshTimer);
      state.forYouDeepRefreshTimer = null;
    }
    state.forYouDeepRefreshLastAt = 0;
    state.playlistsLoaded = false;
    state.profile = {};
    state.likedSongIds = new Set();
    state.dislikedSongIds = new Set();
    state.pendingLikeSongIds.clear();
    state.pendingDislikeSongIds.clear();
    state.playlists.items = [];
    state.playlists.selectedId = DEFAULT_PLAYLIST_ID;
    state.lists.foryou = [];
    state.lists.liked = [];
    state.lists.history = [];
    state.lists.playlists = [];
    state.forYouPaging.tracks = [];
    state.forYouPaging.currentPage = 0;
    state.forYouPaging.loading = false;
    state.forYouPaging.exhausted = false;
    state.shufflePools = new Map();
    syncLikeUI();
    renderForYou();
    renderLikedSongs();
    renderHistoryTracks();
    renderPlaylists();
  }

  async function switchIdentity(userId) {
    const next = String(userId || "").trim();
    if (!next) return;
    if (state.userId === next && state.playlistsLoaded) {
      updateIdentityUI();
      updatePersonalSummary();
      return;
    }
    state.userId = next;
    resetUserScopedStateForIdentity();
    updateIdentityUI();
    updatePersonalSummary();
    renderForYouSkeleton(8);
    setForYouStatus("Loading quick personalized picks...");
    const profilePromise = loadUserProfile(true);
    const playlistsPromise = loadPlaylists(true);
    await loadForYou(true, { initialLoad: true, fastMode: true });
    await Promise.allSettled([profilePromise, playlistsPromise]);
  }

  function openAuthAvatarPicker() {
    if (!state.auth.authenticated || !$.authAvatarInput) return;
    try {
      $.authAvatarInput.value = "";
    } catch (error) {}
    $.authAvatarInput.click();
  }

  function openAuthBannerPicker() {
    if (!state.auth.authenticated || !$.authBannerInput) return;
    try {
      $.authBannerInput.value = "";
    } catch (error) {}
    $.authBannerInput.click();
  }

  async function uploadAccountAvatarFromFile(file) {
    if (!state.auth.authenticated || !state.auth.userId || !file) return false;
    const mime = String(file.type || "").toLowerCase();
    if (!mime.startsWith("image/")) {
      setAuthModalStatus("Choose an image file.", true);
      return false;
    }
    const maxBytes = 1_500_000;
    if (Number(file.size || 0) > maxBytes) {
      setAuthModalStatus("Image is too large. Use a file under 1.5 MB.", true);
      return false;
    }
    setAuthBusy(true);
    setAuthModalStatus("Updating photo...");
    try {
      const dataUrl = await readFileAsDataUrl(file);
      const payload = await requestJSON(
        "/api/auth/avatar",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ avatar_image: dataUrl }),
        },
        { timeoutMs: 12000, retries: 0 },
      );
      const user = payload && payload.user ? payload.user : {};
      state.auth.avatarImage = String(user.avatar_image || dataUrl || "");
      updateIdentityUI();
      renderAuthModal();
      setAuthModalStatus("Profile photo updated.");
      return true;
    } catch (error) {
      setAuthModalStatus(
        friendlyAuthApiError(error, "Could not update profile photo."),
        true,
      );
      return false;
    } finally {
      setAuthBusy(false);
      if ($.authAvatarInput) {
        try {
          $.authAvatarInput.value = "";
        } catch (innerError) {}
      }
    }
  }

  async function uploadAccountBannerFromFile(file) {
    if (!state.auth.authenticated || !state.auth.userId || !file) return false;
    const mime = String(file.type || "").toLowerCase();
    if (!mime.startsWith("image/")) {
      setAuthModalStatus("Choose an image file.", true);
      return false;
    }
    const maxBytes = 2_500_000;
    if (Number(file.size || 0) > maxBytes) {
      setAuthModalStatus("Banner is too large. Use a file under 2.5 MB.", true);
      return false;
    }
    setAuthBusy(true);
    setAuthModalStatus("Updating banner...");
    try {
      const dataUrl = await readFileAsDataUrl(file);
      const payload = await requestJSON(
        "/api/auth/banner",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ banner_image: dataUrl }),
        },
        { timeoutMs: 12000, retries: 0 },
      );
      const user = payload && payload.user ? payload.user : {};
      state.auth.bannerImage = String(user.banner_image || dataUrl || "");
      updateIdentityUI();
      renderAuthModal();
      setAuthModalStatus("Banner updated.");
      return true;
    } catch (error) {
      setAuthModalStatus(
        friendlyAuthApiError(error, "Could not update banner."),
        true,
      );
      return false;
    } finally {
      setAuthBusy(false);
      if ($.authBannerInput) {
        try {
          $.authBannerInput.value = "";
        } catch (innerError) {}
      }
    }
  }

  async function loginWithCredentials() {
    if (state.auth.busy) return false;
    const creds = authFormCredentials();
    if (!creds.username || !creds.password) {
      setAuthModalStatus("Username and password are required.", true);
      return false;
    }
    setAuthBusy(true);
    setAuthModalStatus("Logging in...");
    try {
      await requestJSON(
        "/api/auth/login",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(creds),
        },
        { timeoutMs: 7000, retries: 0 },
      );
      await refreshAuthSession(true);
      if (!state.auth.authenticated || !state.auth.userId) {
        setAuthModalStatus("Login failed.", true);
        return false;
      }
      if ($.authPassword) $.authPassword.value = "";
      closeAuthModal();
      void switchIdentity(state.auth.userId);
      return true;
    } catch (error) {
      setAuthModalStatus(friendlyAuthApiError(error, "Login failed."), true);
      return false;
    } finally {
      setAuthBusy(false);
    }
  }

  async function registerWithCredentials() {
    if (state.auth.busy) return false;
    const creds = authFormCredentials();
    if (!creds.username || !creds.password) {
      setAuthModalStatus("Username and password are required.", true);
      return false;
    }
    setAuthBusy(true);
    setAuthModalStatus("Creating account...");
    try {
      await requestJSON(
        "/api/auth/register",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(creds),
        },
        { timeoutMs: 8000, retries: 0 },
      );
      await refreshAuthSession(true);
      if (!state.auth.authenticated || !state.auth.userId) {
        setAuthModalStatus("Account creation failed.", true);
        return false;
      }
      if ($.authPassword) $.authPassword.value = "";
      closeAuthModal();
      void switchIdentity(state.auth.userId);
      return true;
    } catch (error) {
      setAuthModalStatus(
        friendlyAuthApiError(error, "Could not create account."),
        true,
      );
      return false;
    } finally {
      setAuthBusy(false);
    }
  }

  async function logoutAccount() {
    if (state.auth.busy) return false;
    setAuthBusy(true);
    setAuthModalStatus("Logging out...");
    try {
      await requestJSON(
        "/api/auth/logout",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        },
        { timeoutMs: 4500, retries: 0 },
      );
      await refreshAuthSession(true);
      const guestUserId = getOrCreateUserId();
      await switchIdentity(guestUserId);
      closeAuthModal();
      return true;
    } catch (error) {
      setAuthModalStatus(
        friendlyAuthApiError(error, "Could not logout right now."),
        true,
      );
      return false;
    } finally {
      setAuthBusy(false);
    }
  }

  function setHomeQueueCaption(text) {
    if (!$.homeQueueCaption) return;
    $.homeQueueCaption.textContent =
      String(text || "").trim() ||
      "Search from the landing box to build a continuous queue.";
  }

  function setActiveNav(view) {
    $.navItems.forEach(function (item) {
      const isActive = String(item.dataset.view || "") === String(view || "");
      item.classList.toggle("active", isActive);
    });
  }

  function isSettingsMenuOpen() {
    return Boolean(
      $.settingsMenu &&
        state.settingsMenu.open &&
        !$.settingsMenu.classList.contains("hidden"),
    );
  }

  function closeSettingsMenu() {
    if (!$.settingsMenu) return;
    $.settingsMenu.classList.add("hidden");
    $.settingsMenu.setAttribute("aria-hidden", "true");
    state.settingsMenu.open = false;
    state.settingsMenu.anchorButton = null;
    $.settingsButtons.forEach(function (button) {
      if (!button) return;
      button.setAttribute("aria-expanded", "false");
    });
  }

  function positionSettingsMenu(anchorButton) {
    if (!$.settingsMenu || !anchorButton) return;
    const menu = $.settingsMenu;
    const buttonRect = anchorButton.getBoundingClientRect();
    const menuRect = menu.getBoundingClientRect();
    const viewportWidth =
      typeof window.innerWidth === "number" ? window.innerWidth : 0;
    const viewportHeight =
      typeof window.innerHeight === "number" ? window.innerHeight : 0;
    const margin = 12;
    const gap = 10;

    let left = buttonRect.right - menuRect.width;
    if (left < margin) left = margin;
    if (left + menuRect.width > viewportWidth - margin) {
      left = Math.max(margin, viewportWidth - menuRect.width - margin);
    }

    let top = buttonRect.bottom + gap;
    if (top + menuRect.height > viewportHeight - margin) {
      top = Math.max(margin, buttonRect.top - menuRect.height - gap);
    }

    menu.style.left = `${Math.round(left)}px`;
    menu.style.top = `${Math.round(top)}px`;
  }

  function openSettingsMenu(anchorButton) {
    if (!$.settingsMenu || !anchorButton) return;
    state.settingsMenu.open = true;
    state.settingsMenu.anchorButton = anchorButton;
    $.settingsMenu.classList.remove("hidden");
    $.settingsMenu.setAttribute("aria-hidden", "false");
    $.settingsButtons.forEach(function (button) {
      if (!button) return;
      button.setAttribute("aria-expanded", button === anchorButton ? "true" : "false");
    });
    positionSettingsMenu(anchorButton);
  }

  function toggleSettingsMenu(anchorButton) {
    if (!anchorButton) return;
    const sameAnchor = state.settingsMenu.anchorButton === anchorButton;
    if (isSettingsMenuOpen() && sameAnchor) {
      closeSettingsMenu();
      return;
    }
    openSettingsMenu(anchorButton);
  }

  function syncHomePlaybackChromeVisibility() {
    const onHome = String(state.activeView || "") === "home";
    const hasTrack = Boolean(
      state.current &&
        (state.current.trackId || state.current.videoId || state.current.title),
    );
    const showHomePlaybackChrome = onHome && hasTrack;
    try {
      document.body.classList.toggle(
        "home-track-ui-visible",
        showHomePlaybackChrome,
      );
      // Keep right-side "Now Playing" collapsed until a real track exists.
      document.body.classList.toggle("now-playing-visible", hasTrack);
    } catch (error) {}

    if ($.player) {
      const hideFloatingPlayer = onHome && !showHomePlaybackChrome;
      $.player.classList.toggle("view-hidden", hideFloatingPlayer);
      $.player.setAttribute("aria-hidden", hideFloatingPlayer ? "true" : "false");
    }
  }

  function switchView(view) {
    const next = String(view || "").trim() || "home";
    state.activeView = next;
    try {
      document.body.dataset.appView = next;
    } catch (error) {}
    syncHomePlaybackChromeVisibility();
    setActiveNav(next);
    $.viewPanels.forEach(function (panel) {
      const isActive = String(panel.dataset.viewPanel || "") === next;
      panel.classList.toggle("active", isActive);
      if (isActive) {
        panel.classList.remove("view-enter");
        // reflow to replay transition animation
        void panel.offsetWidth;
        panel.classList.add("view-enter");
      }
    });
    if (next === "foryou") {
      setTimeout(function () {
        const resized = updateForYouPageSize(true);
        const pageSize = Math.max(
          1,
          Number(state.forYouPaging.pageSize) || FOR_YOU_PAGE_SIZE,
        );
        const poolCount = Array.isArray(state.forYouPaging.tracks)
          ? state.forYouPaging.tracks.length
          : 0;
        if (!poolCount) {
          void loadForYou(false);
          return;
        }
        if (resized) renderForYou();
        else updateForYouStripControls();
      }, 80);
    }
  }

  function sleep(ms) {
    return new Promise(function (resolve) {
      setTimeout(resolve, Number(ms) || 0);
    });
  }

  function stripFeatureSuffix(value) {
    return String(value || "")
      .replace(/\s*[\(\[]?\s*(feat\.?|featuring|ft\.)\s+[^\)\]]+[\)\]]?$/i, "")
      .trim();
  }

  function normalizeKeyPart(value) {
    return stripFeatureSuffix(value)
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
  }

  function isYoutubeId(value) {
    return /^[A-Za-z0-9_-]{11}$/.test(String(value || ""));
  }

  function isEmbedBlockedYoutubeErrorCode(code) {
    return Number(code) === 150 || Number(code) === 101;
  }

  function pruneBlockedVideoIdCache() {
    const now = Date.now();
    state.ytBlockedVideoIds.forEach(function (meta, videoId) {
      const ts = Number(meta && meta.ts);
      if (!Number.isFinite(ts) || now - ts > YT_EMBED_BLOCK_TTL_MS) {
        state.ytBlockedVideoIds.delete(videoId);
      }
    });
  }

  function markVideoIdEmbedBlocked(videoId, code) {
    const id = String(videoId || "").trim();
    if (!isYoutubeId(id)) return;
    pruneBlockedVideoIdCache();
    state.ytBlockedVideoIds.set(id, {
      code: Number(code) || 150,
      ts: Date.now(),
    });
  }

  function isVideoIdEmbedBlocked(videoId) {
    const id = String(videoId || "").trim();
    if (!isYoutubeId(id)) return false;
    pruneBlockedVideoIdCache();
    return state.ytBlockedVideoIds.has(id);
  }

  function chooseYoutubeId(candidates) {
    for (let i = 0; i < candidates.length; i += 1) {
      const candidate = String(candidates[i] || "").trim();
      if (isYoutubeId(candidate)) return candidate;
    }
    return "";
  }

  function youtubeThumb(videoId, quality) {
    if (!isYoutubeId(videoId)) return "";
    return `https://i.ytimg.com/vi/${videoId}/${quality}.jpg`;
  }

  function isYoutubeThumbUrl(value) {
    return /(^https?:)?\/\/i\.ytimg\.com\/vi\//i.test(String(value || "").trim());
  }

  function isSpotifyImageUrl(value) {
    return /(^https?:)?\/\/i\.scdn\.co\/image\//i.test(String(value || "").trim());
  }

  function isLocalFallbackCoverUrl(value) {
    return /cover-fallback\.svg(?:\?|$)/i.test(String(value || "").trim());
  }

  function cssUrlValue(value) {
    const safe = String(value || "").replace(/"/g, '\\"');
    return `url("${safe}")`;
  }

  function youtubeThumbQualityRank(value) {
    const url = String(value || "").toLowerCase();
    if (!isYoutubeThumbUrl(url)) return -1;
    if (url.includes("/maxresdefault.")) return 60;
    if (url.includes("/hq720.")) return 55;
    if (url.includes("/sddefault.")) return 50;
    if (url.includes("/hqdefault.")) return 40;
    if (url.includes("/mqdefault.")) return 30;
    if (url.includes("/default.")) return 20;
    return 10;
  }

  function isLikelyPlaceholderCoverUrl(value) {
    const url = String(value || "").trim().toLowerCase();
    if (!url) return true;
    return (
      url.includes("placehold.co") ||
      url.includes("via.placeholder.com") ||
      url.includes("dummyimage.com") ||
      url.includes("/no-image") ||
      url.includes("/noimage") ||
      url.includes("/placeholder")
    );
  }

  function cleanCoverCandidates(values) {
    return uniq(values).filter(function (value) {
      return !isLikelyPlaceholderCoverUrl(value) && !isLocalFallbackCoverUrl(value);
    });
  }

  function prioritizeCoverCandidates(values) {
    const cleaned = cleanCoverCandidates(values);
    const spotify = [];
    const nonYoutube = [];
    const youtube = [];
    cleaned.forEach(function (url) {
      if (isYoutubeThumbUrl(url)) youtube.push(url);
      else if (isSpotifyImageUrl(url)) spotify.push(url);
      else nonYoutube.push(url);
    });
    youtube.sort(function (a, b) {
      return youtubeThumbQualityRank(b) - youtubeThumbQualityRank(a);
    });
    return [...spotify, ...nonYoutube, ...youtube];
  }

  function uniq(values) {
    const seen = new Set();
    const out = [];
    (values || []).forEach(function (value) {
      if (!value) return;
      if (seen.has(value)) return;
      seen.add(value);
      out.push(value);
    });
    return out;
  }

  function toOptionalBool(value) {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return Boolean(value);
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      if (["1", "true", "yes", "y"].includes(normalized)) return true;
      if (["0", "false", "no", "n"].includes(normalized)) return false;
    }
    return null;
  }

  function shortDescription(value) {
    const text = String(value || "")
      .replace(/\s+/g, " ")
      .trim();
    if (!text) return "";
    if (text.length <= 220) return text;
    const clipped = text.slice(0, 220);
    const safe = clipped.includes(" ")
      ? clipped.slice(0, clipped.lastIndexOf(" "))
      : clipped;
    return `${safe}...`;
  }

  function formatTime(seconds) {
    const s = Math.max(0, Number(seconds || 0));
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return `${m}:${r < 10 ? "0" : ""}${r}`;
  }

  function updateRangeFill(el) {
    if (!el) return;
    el.style.setProperty("--progress", `${el.value}%`);
  }

  function listFor(key) {
    if (key === "playlists") return state.lists.playlists;
    if (key === "liked") return state.lists.liked;
    if (key === "history") return state.lists.history;
    if (key === "foryou") return state.lists.foryou;
    if (key === "home") return state.lists.home;
    return state.lists.search;
  }

  function songIdForTrack(track) {
    if (!track) return "";
    const trackId = String(track.trackId || "").trim();
    if (trackId) return trackId;
    const videoId = String(track.videoId || "").trim();
    return videoId;
  }

  function trackCacheKey(track) {
    return [
      String(track && track.trackId ? track.trackId : ""),
      String(track && track.videoId ? track.videoId : ""),
      String(track && track.title ? track.title : ""),
      String(track && track.artist ? track.artist : ""),
    ].join("|");
  }

  function lyricsCacheKeyForTrack(track) {
    if (!track) return "";
    const title = normalizeKeyPart(track.title || "");
    const artist = normalizeKeyPart(track.artist || "");
    if (!title && !artist) return "";
    return `${artist}|${title}`;
  }

  function shouldPrefetchLyricsForTrack(track) {
    if (!track) return false;
    const key = lyricsCacheKeyForTrack(track);
    if (!key) return false;
    if (state.current && lyricsCacheKeyForTrack(state.current) === key) return false;
    if (state.lyricsPrefetchPending.has(key)) return false;
    const doneAt = Number(state.lyricsPrefetchDoneAt.get(key) || 0);
    if (doneAt && Date.now() - doneAt < LYRICS_PREFETCH_MARK_TTL_MS) return false;
    return true;
  }

  async function prefetchLyricsForTrack(track) {
    const key = lyricsCacheKeyForTrack(track);
    if (!key || !shouldPrefetchLyricsForTrack(track)) return;
    state.lyricsPrefetchPending.add(key);
    try {
      const params = new URLSearchParams({
        title: track.title || "",
        artist: track.artist || "",
        prefetch: "1",
      });
      await requestJSON(
        `/api/lyrics?${params.toString()}`,
        {},
        { timeoutMs: LYRICS_PREFETCH_TIMEOUT_MS, retries: 0 },
      );
      state.lyricsPrefetchDoneAt.set(key, Date.now());
    } catch (error) {
      // Intentionally ignore prefetch failures; on-demand load still handles retries/timeouts.
    } finally {
      state.lyricsPrefetchPending.delete(key);
      drainLyricsPrefetchQueue();
    }
  }

  function drainLyricsPrefetchQueue() {
    while (
      state.lyricsPrefetchActive < LYRICS_PREFETCH_CONCURRENCY &&
      state.lyricsPrefetchQueue.length
    ) {
      const nextTrack = state.lyricsPrefetchQueue.shift();
      if (!shouldPrefetchLyricsForTrack(nextTrack)) {
        continue;
      }
      state.lyricsPrefetchActive += 1;
      prefetchLyricsForTrack(nextTrack).finally(function () {
        state.lyricsPrefetchActive = Math.max(0, state.lyricsPrefetchActive - 1);
        drainLyricsPrefetchQueue();
      });
    }
  }

  function queueLyricsPrefetchForVisibleTracks(tracks, limit) {
    const max = Math.max(0, Number(limit) || 0);
    if (!max) return;
    const list = Array.isArray(tracks) ? tracks : [];
    if (!list.length) return;

    const now = Date.now();
    state.lyricsPrefetchDoneAt.forEach(function (ts, key) {
      if (now - Number(ts || 0) > LYRICS_PREFETCH_MARK_TTL_MS) {
        state.lyricsPrefetchDoneAt.delete(key);
      }
    });

    let added = 0;
    for (let i = 0; i < list.length && added < max; i += 1) {
      const track = list[i];
      if (!shouldPrefetchLyricsForTrack(track)) continue;
      state.lyricsPrefetchQueue.push(track);
      added += 1;
    }

    if (added && !state.lyricsPrefetchTimer) {
      state.lyricsPrefetchTimer = setTimeout(function () {
        state.lyricsPrefetchTimer = null;
        drainLyricsPrefetchQueue();
      }, LYRICS_PREFETCH_DEFER_MS);
    }
  }

  function parseBooleanFlag(value) {
    const raw = String(value == null ? "" : value).trim().toLowerCase();
    return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
  }

  function getOrCreateUserId() {
    const bodySharedUserId = String(
      (document.body && document.body.getAttribute("data-eraex-shared-user-id")) || "",
    ).trim();
    const bodyForceSharedUserId = parseBooleanFlag(
      document.body && document.body.getAttribute("data-eraex-force-shared-user-id"),
    );
    const legacySharedUserId = String(window.ERAEX_SHARED_USER_ID || "").trim();
    const legacyForceSharedUserId = parseBooleanFlag(window.ERAEX_FORCE_SHARED_USER_ID);
    const sharedUserId = bodySharedUserId || legacySharedUserId;
    const forceSharedUserId = bodyForceSharedUserId || legacyForceSharedUserId;

    if (sharedUserId && forceSharedUserId) {
      localStorage.setItem(USER_ID_KEY, sharedUserId);
      return sharedUserId;
    }

    let userId = String(localStorage.getItem(USER_ID_KEY) || "").trim();
    if (!userId && sharedUserId) {
      userId = sharedUserId;
      localStorage.setItem(USER_ID_KEY, userId);
      return userId;
    }
    if (!userId) {
      if (window.crypto && typeof window.crypto.randomUUID === "function") {
        userId = window.crypto.randomUUID();
      } else {
        userId = `u_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
      }
      localStorage.setItem(USER_ID_KEY, userId);
    }
    return userId;
  }

  function readRepeatOnePreference() {
    try {
      const raw = String(localStorage.getItem(REPEAT_ONE_STORAGE_KEY) || "")
        .trim()
        .toLowerCase();
      return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
    } catch (error) {
      return false;
    }
  }

  function persistRepeatOnePreference(enabled) {
    try {
      localStorage.setItem(REPEAT_ONE_STORAGE_KEY, enabled ? "1" : "0");
    } catch (error) {}
  }

  function syncRepeatButtonUI() {
    if (!$.repeatBtn) return;
    const on = !!state.repeatOne;
    $.repeatBtn.classList.toggle("is-active", on);
    $.repeatBtn.setAttribute("aria-pressed", on ? "true" : "false");
    $.repeatBtn.setAttribute("data-tooltip", on ? "Repeat one on" : "Repeat one");
  }

  function setRepeatOneEnabled(enabled) {
    state.repeatOne = !!enabled;
    persistRepeatOnePreference(state.repeatOne);
    syncRepeatButtonUI();
  }

  function readShufflePreference() {
    try {
      const raw = String(localStorage.getItem(SHUFFLE_STORAGE_KEY) || "")
        .trim()
        .toLowerCase();
      return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
    } catch (error) {
      return false;
    }
  }

  function persistShufflePreference(enabled) {
    try {
      localStorage.setItem(SHUFFLE_STORAGE_KEY, enabled ? "1" : "0");
    } catch (error) {}
  }

  function shuffleIndices(total) {
    const size = Math.max(0, Number(total) || 0);
    const values = [];
    for (let i = 0; i < size; i += 1) values.push(i);
    for (let i = values.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = values[i];
      values[i] = values[j];
      values[j] = tmp;
    }
    return values;
  }

  function buildShufflePool(total, anchorIndex) {
    const size = Math.max(0, Number(total) || 0);
    const order = shuffleIndices(size);
    const anchor = Number.isInteger(anchorIndex) ? anchorIndex : -1;
    if (anchor >= 0 && anchor < size) {
      const pos = order.indexOf(anchor);
      if (pos > 0) {
        const tmp = order[0];
        order[0] = order[pos];
        order[pos] = tmp;
      }
    }
    return {
      order,
      pointer: anchor >= 0 && anchor < size ? 0 : -1,
      size,
    };
  }

  function getShufflePool(listKey, total, currentIndex) {
    const key = String(listKey || "").trim();
    const size = Math.max(0, Number(total) || 0);
    if (!key || size <= 0) return null;

    let pool = state.shufflePools.get(key);
    const needsReset =
      !pool ||
      !Array.isArray(pool.order) ||
      Number(pool.size || 0) !== size ||
      pool.order.length !== size;
    if (needsReset) {
      pool = buildShufflePool(size, currentIndex);
      state.shufflePools.set(key, pool);
    }
    const safeIndex = Number.isInteger(currentIndex) ? currentIndex : -1;
    if (safeIndex >= 0 && safeIndex < size) {
      const pos = pool.order.indexOf(safeIndex);
      if (pos >= 0) pool.pointer = pos;
    }
    return pool;
  }

  function syncShuffleButtonUI() {
    if (!$.shuffleBtn) return;
    const on = !!state.shuffleEnabled;
    $.shuffleBtn.classList.toggle("is-active", on);
    $.shuffleBtn.setAttribute("aria-pressed", on ? "true" : "false");
    $.shuffleBtn.setAttribute("data-tooltip", on ? "Shuffle on" : "Shuffle");
  }

  function setShuffleEnabled(enabled) {
    state.shuffleEnabled = !!enabled;
    persistShufflePreference(state.shuffleEnabled);
    state.shufflePools = new Map();
    syncShuffleButtonUI();
    const list = listFor(state.activeListKey);
    if (state.shuffleEnabled && Array.isArray(list) && list.length > 0) {
      getShufflePool(state.activeListKey, list.length, state.queueIndex);
    }
  }

  function nextShuffleIndex(listKey, total, currentIndex) {
    const size = Math.max(0, Number(total) || 0);
    if (size <= 0) return null;
    if (size === 1) return 0;
    const current = Number.isInteger(currentIndex) ? currentIndex : -1;
    const pool = getShufflePool(listKey, size, current);
    if (!pool) return null;

    let pointer = Number(pool.pointer);
    if (!Number.isInteger(pointer) || pointer < 0) pointer = pool.order.indexOf(current);
    if (!Number.isInteger(pointer) || pointer < 0) pointer = 0;

    let nextPointer = pointer + 1;
    if (nextPointer >= pool.order.length) {
      const refreshed = buildShufflePool(size, current);
      state.shufflePools.set(String(listKey || "").trim(), refreshed);
      pool.order = refreshed.order;
      pool.pointer = refreshed.pointer;
      nextPointer = pool.order.length > 1 ? 1 : 0;
    }
    pool.pointer = nextPointer;
    return Number(pool.order[nextPointer]);
  }

  function previousShuffleIndex(listKey, total, currentIndex) {
    const size = Math.max(0, Number(total) || 0);
    if (size <= 0) return null;
    const pool = getShufflePool(listKey, size, currentIndex);
    if (!pool) return null;
    let pointer = Number(pool.pointer);
    if (!Number.isInteger(pointer) || pointer < 0) return null;
    if (pointer <= 0) return null;
    pointer -= 1;
    pool.pointer = pointer;
    return Number(pool.order[pointer]);
  }

  async function fetchWithTimeout(url, options, timeoutMs) {
    const timeout = Number(timeoutMs || 0);
    if (!timeout || timeout <= 0) return fetch(url, options);

    const sourceSignal = options && options.signal;
    const controller = new AbortController();
    if (sourceSignal && typeof sourceSignal.addEventListener === "function") {
      if (sourceSignal.aborted) {
        controller.abort();
      } else {
        sourceSignal.addEventListener(
          "abort",
          function () {
            controller.abort();
          },
          { once: true },
        );
      }
    }

    const timer = setTimeout(function () {
      controller.abort();
    }, timeout);

    try {
      const nextOptions = { ...(options || {}) };
      delete nextOptions.signal;
      return await fetch(url, {
        ...nextOptions,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  async function requestJSON(url, options, config) {
    const cfg = config || {};
    const timeoutMs = Number(cfg.timeoutMs || 8000);
    const retries = Math.max(0, Number(cfg.retries || 0));
    const backoffMs = Math.max(120, Number(cfg.backoffMs || 260));

    let attempt = 0;
    let lastError = null;
    while (attempt <= retries) {
      try {
        const response = await fetchWithTimeout(url, options, timeoutMs);
        const text = await response.text();
        const payload = text
          ? (() => {
              try {
                return JSON.parse(text);
              } catch (error) {
                return {};
              }
            })()
          : {};

        if (response.ok) return payload;

        const retryable = response.status >= 500 || response.status === 429;
        if (retryable && attempt < retries) {
          attempt += 1;
          await sleep(backoffMs * attempt);
          continue;
        }

        const rawText = String(text || "").trim();
        const looksHtml = /^<!doctype|^<html/i.test(rawText);
        let message = payload.error || "";
        if (!message && rawText && !looksHtml) {
          message = rawText;
        }
        if (!message) {
          if (looksHtml && response.status === 404) {
            message = "API route not found (404).";
          } else {
            message = `Request failed with status ${response.status}`;
          }
        }
        const requestError = new Error(String(message).slice(0, 220));
        requestError.status = response.status;
        requestError.isHtml = looksHtml;
        throw requestError;
      } catch (error) {
        if (error.name === "AbortError") throw error;
        lastError = error;
        if (attempt >= retries) break;
        attempt += 1;
        await sleep(backoffMs * attempt);
      }
    }
    throw lastError || new Error("Request failed");
  }

  function postPlayerDebug(eventName, payload) {
    try {
      const body = JSON.stringify({
        event: String(eventName || ""),
        ...(payload || {}),
      });
      if (navigator && typeof navigator.sendBeacon === "function") {
        const blob = new Blob([body], { type: "application/json" });
        navigator.sendBeacon("/api/player_debug", blob);
        return;
      }
      void fetch("/api/player_debug", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        keepalive: true,
      }).catch(function () {});
    } catch (error) {
      // Best-effort debug only.
    }
  }

  function spotifyEnabled() {
    return !!(state.spotify && state.spotify.enabled);
  }

  function spotifyAuthenticated() {
    return !!(state.spotify && state.spotify.authenticated);
  }

  function setPlaybackProvider(provider) {
    state.activePlaybackProvider = provider === "spotify" ? "spotify" : "youtube";
  }

  async function refreshSpotifyStatus() {
    try {
      const data = await requestJSON("/api/spotify/status", {}, { timeoutMs: 5000, retries: 0 });
      state.spotify.statusKnown = true;
      state.spotify.enabled = !!data.enabled;
      state.spotify.authenticated = !!data.authenticated;
      return data;
    } catch (error) {
      state.spotify.statusKnown = true;
      state.spotify.enabled = false;
      state.spotify.authenticated = false;
      return null;
    }
  }

  async function getSpotifyAccessToken() {
    const now = Date.now() / 1000;
    if (
      state.spotify.token &&
      Number.isFinite(state.spotify.tokenExpiresAt) &&
      state.spotify.tokenExpiresAt - now > 30
    ) {
      return state.spotify.token;
    }
    const data = await requestJSON("/api/spotify/access_token", {}, { timeoutMs: 6000, retries: 0 });
    const token = String(data.access_token || "").trim();
    if (!token) throw new Error("Spotify auth required");
    state.spotify.token = token;
    state.spotify.tokenExpiresAt = Number(data.expires_at || 0);
    state.spotify.authenticated = true;
    return token;
  }

  async function spotifyApiFetch(path, options) {
    const token = await getSpotifyAccessToken();
    const opts = options || {};
    const headers = { ...(opts.headers || {}), Authorization: `Bearer ${token}` };
    return fetchWithTimeout(path, { ...opts, headers }, Number(opts.timeoutMs || 8000));
  }

  function loadSpotifySDKScript() {
    if (state.spotify.sdkLoadPromise) return state.spotify.sdkLoadPromise;
    state.spotify.sdkLoadPromise = new Promise(function (resolve, reject) {
      if (window.Spotify && window.Spotify.Player) {
        state.spotify.sdkScriptLoaded = true;
        resolve();
        return;
      }
      const existing = document.querySelector('script[data-eraex-spotify-sdk=\"1\"]');
      if (existing) {
        const prev = window.onSpotifyWebPlaybackSDKReady;
        window.onSpotifyWebPlaybackSDKReady = function () {
          state.spotify.sdkScriptLoaded = true;
          if (typeof prev === "function") {
            try {
              prev();
            } catch (e) {}
          }
          resolve();
        };
        setTimeout(function () {
          if (window.Spotify && window.Spotify.Player) resolve();
        }, 200);
        return;
      }
      const prev = window.onSpotifyWebPlaybackSDKReady;
      window.onSpotifyWebPlaybackSDKReady = function () {
        state.spotify.sdkScriptLoaded = true;
        if (typeof prev === "function") {
          try {
            prev();
          } catch (e) {}
        }
        resolve();
      };
      const script = document.createElement("script");
      script.src = "https://sdk.scdn.co/spotify-player.js";
      script.async = true;
      script.dataset.eraexSpotifySdk = "1";
      script.onerror = function () {
        reject(new Error("Failed to load Spotify SDK"));
      };
      document.head.appendChild(script);
    });
    return state.spotify.sdkLoadPromise;
  }

  function updateSpotifyProgressSnapshotFromState(sdkState) {
    if (!sdkState) {
      state.spotify.playbackState = null;
      state.spotify.progressSnapshot = null;
      return;
    }
    state.spotify.playbackState = sdkState;
    state.spotify.progressSnapshot = {
      positionMs: Number(sdkState.position || 0),
      durationMs: Number((sdkState.duration || (sdkState.track_window && sdkState.track_window.current_track && sdkState.track_window.current_track.duration_ms)) || 0),
      paused: !!sdkState.paused,
      updatedAt: Date.now(),
      trackUri:
        (sdkState.track_window &&
          sdkState.track_window.current_track &&
          sdkState.track_window.current_track.uri) ||
        "",
    };
  }

  function handleSpotifyPlayerStateChanged(sdkState) {
    const prev = state.spotify.progressSnapshot;
    updateSpotifyProgressSnapshotFromState(sdkState);
    if (!sdkState) return;
    const isPaused = !!sdkState.paused;
    const currentUri =
      (sdkState.track_window &&
        sdkState.track_window.current_track &&
        sdkState.track_window.current_track.uri) ||
      "";

    // Ignore background Spotify pause/idle events when YouTube is the active provider.
    // We still allow Spotify "playing" events below so fallback takeover can switch providers.
    if (isPaused && state.activePlaybackProvider !== "spotify") {
      return;
    }

    if (!isPaused) {
      setPlaybackProvider("spotify");
      setPlayState(true);
      startProgress();
      updateProgress(true);
      void recordCurrentPlayOnce();
      postPlayerDebug("spotify_state_playing", {
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId || "",
            }
          : null,
        note: currentUri,
      });
      return;
    }

    setPlayState(false);
    updateProgress(true);
    const prevWasPlaying = !!(prev && prev.paused === false);
    const endedHeuristic =
      prevWasPlaying &&
      Number(sdkState.position || 0) === 0 &&
      !!prev &&
      String(prev.trackUri || "") === String(currentUri || "");
    if (endedHeuristic) {
      if (state.repeatOne) {
        void replayCurrentTrackAfterEnd("spotify");
        return;
      }
      stopProgress();
      playNext(true);
      return;
    }
    stopProgress();
  }

  async function ensureSpotifyPlayerReady() {
    if (!spotifyEnabled()) return false;
    if (!spotifyAuthenticated()) {
      await refreshSpotifyStatus();
      if (!spotifyAuthenticated()) return false;
    }
    await loadSpotifySDKScript();
    if (state.spotify.playerReady && state.spotify.player) return true;
    if (state.spotify.playerReadyPromise) return state.spotify.playerReadyPromise;
    state.spotify.playerReadyPromise = (async function () {
      const token = await getSpotifyAccessToken();
      if (!window.Spotify || !window.Spotify.Player) {
        throw new Error("Spotify SDK unavailable");
      }
      const player =
        state.spotify.player ||
        new window.Spotify.Player({
          name: "EraEx Fallback Player",
          getOAuthToken: function (cb) {
            getSpotifyAccessToken()
              .then(function (t) {
                cb(t);
              })
              .catch(function () {
                cb("");
              });
          },
          volume: ($.volume ? Number($.volume.value || 100) : 100) / 100,
        });
      state.spotify.player = player;

      if (!player.__eraexBound) {
        player.__eraexBound = true;
        player.addListener("ready", function ({ device_id }) {
          state.spotify.playerReady = true;
          state.spotify.playerDeviceId = String(device_id || "");
          postPlayerDebug("spotify_ready", { note: state.spotify.playerDeviceId });
        });
        player.addListener("not_ready", function ({ device_id }) {
          state.spotify.playerReady = false;
          if (String(state.spotify.playerDeviceId || "") === String(device_id || "")) {
            state.spotify.playerDeviceId = "";
          }
          postPlayerDebug("spotify_not_ready", { note: String(device_id || "") });
        });
        player.addListener("player_state_changed", handleSpotifyPlayerStateChanged);
        player.addListener("initialization_error", function (e) {
          postPlayerDebug("spotify_init_error", { note: (e && e.message) || "" });
        });
        player.addListener("authentication_error", function (e) {
          state.spotify.authenticated = false;
          postPlayerDebug("spotify_auth_error", { note: (e && e.message) || "" });
        });
        player.addListener("account_error", function (e) {
          postPlayerDebug("spotify_account_error", { note: (e && e.message) || "" });
        });
        player.addListener("playback_error", function (e) {
          postPlayerDebug("spotify_playback_error", { note: (e && e.message) || "" });
        });
      }

      if (!state.spotify.playerConnecting) {
        state.spotify.playerConnecting = true;
        try {
          await player.connect();
        } finally {
          state.spotify.playerConnecting = false;
        }
      }

      const started = Date.now();
      while (!state.spotify.playerReady && Date.now() - started < 12000) {
        await sleep(120);
      }
      if (!state.spotify.playerReady || !state.spotify.playerDeviceId) {
        throw new Error("Spotify player device not ready");
      }
      // Ensure token callback path is healthy.
      if (!token) throw new Error("Spotify token unavailable");
      return true;
    })()
      .catch(function (error) {
        state.spotify.playerReadyPromise = null;
        throw error;
      })
      .then(function (ok) {
        state.spotify.playerReadyPromise = null;
        return ok;
      });
    return state.spotify.playerReadyPromise;
  }

  async function openSpotifyLoginPopup() {
    if (state.spotify.authPopup && !state.spotify.authPopup.closed) {
      try {
        state.spotify.authPopup.focus();
      } catch (error) {}
      return true;
    }
    const info = await requestJSON("/api/spotify/login", {}, { timeoutMs: 6000, retries: 0 });
    if (!info || !info.authorize_url) throw new Error("Spotify authorize URL unavailable");
    const popup = window.open(
      info.authorize_url,
      "eraex-spotify-auth",
      "width=520,height=760,resizable=yes,scrollbars=yes",
    );
    if (!popup) throw new Error("Popup blocked");
    state.spotify.authPopup = popup;
    if (state.spotify.authPollTimer) clearInterval(state.spotify.authPollTimer);
    state.spotify.authPollTimer = setInterval(function () {
      if (!state.spotify.authPopup || state.spotify.authPopup.closed) {
        clearInterval(state.spotify.authPollTimer);
        state.spotify.authPollTimer = null;
        void (async function () {
          const status = await refreshSpotifyStatus();
          if (status && status.authenticated) {
            void retryPendingSpotifyFallbackAfterAuth("poll");
          }
        })();
      }
    }, SPOTIFY_STATUS_POLL_MS);
    return true;
  }

  async function ensureSpotifyAuthInteractive() {
    const status = await refreshSpotifyStatus();
    if (!status || !status.enabled) throw new Error("Spotify playback is not configured");
    if (status.authenticated) return true;
    await openSpotifyLoginPopup();
    throw new Error("Complete Spotify login in the popup, then retry playback.");
  }

  async function retryPendingSpotifyFallbackAfterAuth(sourceTag) {
    const pendingKey = String(state.spotify.pendingAuthRetryTrackKey || "").trim();
    if (!pendingKey) return false;
    if (!state.current) {
      state.spotify.pendingAuthRetryTrackKey = "";
      state.spotify.pendingAuthRetryReasonCode = null;
      return false;
    }
    const currentKey = trackCacheKey(state.current);
    if (!currentKey || currentKey !== pendingKey) {
      state.spotify.pendingAuthRetryTrackKey = "";
      state.spotify.pendingAuthRetryReasonCode = null;
      return false;
    }
    if (!spotifyAuthenticated()) return false;

    const reasonCode = state.spotify.pendingAuthRetryReasonCode;
    state.spotify.pendingAuthRetryTrackKey = "";
    state.spotify.pendingAuthRetryReasonCode = null;
    setStatus("Retrying with Spotify fallback...");

    const ok = await trySpotifyFallbackPlayback(reasonCode);
    postPlayerDebug(
      ok ? "spotify_fallback_auth_resume_ok" : "spotify_fallback_auth_resume_fail",
      {
        note: String(sourceTag || ""),
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId || "",
            }
          : null,
      },
    );
    if (!ok) {
      setStatus("Spotify fallback is ready, but playback could not start for this track.", true);
      return false;
    }
    setStatus("");
    return true;
  }

  async function attemptSpotifyFallbackAfterYoutubeFailure(reasonCode) {
    const ok = await trySpotifyFallbackPlayback(reasonCode);
    if (ok) return true;

    let status = null;
    try {
      status = await refreshSpotifyStatus();
    } catch (error) {
      status = null;
    }
    if (!status || !status.enabled || status.authenticated || !state.current) {
      return false;
    }

    const pendingKey = trackCacheKey(state.current);
    if (!pendingKey) return false;
    state.spotify.pendingAuthRetryTrackKey = pendingKey;
    state.spotify.pendingAuthRetryReasonCode = Number.isFinite(Number(reasonCode))
      ? Number(reasonCode)
      : null;

    try {
      await openSpotifyLoginPopup();
      setStatus("YouTube playback failed. Complete Spotify login in the popup to continue playback.");
      postPlayerDebug("spotify_fallback_auth_prompt", {
        code: Number.isFinite(Number(reasonCode)) ? Number(reasonCode) : null,
        track: {
          title: state.current.title,
          artist: state.current.artist,
          videoId: state.current.videoId || "",
        },
      });
      return true;
    } catch (error) {
      postPlayerDebug("spotify_fallback_auth_prompt_fail", {
        code: Number.isFinite(Number(reasonCode)) ? Number(reasonCode) : null,
        note: String((error && error.message) || error || ""),
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId || "",
            }
          : null,
      });
      return false;
    }
  }

  async function trySpotifyFallbackPlayback(reasonCode) {
    if (!state.current || state.spotify.fallbackInFlight) return false;
    if (!spotifyEnabled()) {
      await refreshSpotifyStatus();
      if (!spotifyEnabled()) return false;
    }
    if (!spotifyAuthenticated()) return false;
    state.spotify.fallbackInFlight = true;
    try {
      await ensureSpotifyPlayerReady();
      if (state.spotify.player && typeof state.spotify.player.activateElement === "function") {
        try {
          await state.spotify.player.activateElement();
        } catch (error) {}
      }
      const params = new URLSearchParams({
        track_id: state.current.trackId || state.current.videoId || "",
        title: state.current.title || "",
        artist: state.current.artist || "",
      });
      const resolved = await requestJSON(
        `/api/spotify/resolve_track?${params.toString()}`,
        {},
        { timeoutMs: 8000, retries: 0 },
      );
      const uri = String(resolved.spotify_uri || "").trim();
      if (!uri) return false;
      const deviceId = String(state.spotify.playerDeviceId || "").trim();
      if (!deviceId) return false;
      const resp = await spotifyApiFetch(
        `https://api.spotify.com/v1/me/player/play?device_id=${encodeURIComponent(deviceId)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ uris: [uri] }),
          timeoutMs: 10000,
        },
      );
      if (!resp.ok && resp.status !== 204) {
        const text = await resp.text().catch(function () { return ""; });
        throw new Error(text || `Spotify play failed (${resp.status})`);
      }
      state.current.spotifyUri = uri;
      state.current.spotifyTrackId = String(resolved.spotify_track_id || "");
      state.current.playbackProvider = "spotify";
      if (Array.isArray(resolved.thumbnail_candidates) && resolved.thumbnail_candidates.length) {
        state.current.coverCandidates = prioritizeCoverCandidates([
          ...resolved.thumbnail_candidates,
          ...(state.current.coverCandidates || []),
        ]);
        setImgWithFallback($.playerArt, state.current.coverCandidates, PLAYER_PLACEHOLDER, {
          track: state.current,
          allowYtDlpResolve: false,
        });
        setImgWithFallback($.rightPlayerArt, state.current.coverCandidates, PLAYER_PLACEHOLDER, {
          track: state.current,
          allowYtDlpResolve: false,
        });
      }
      if (state.yt && typeof state.yt.stopVideo === "function") {
        try {
          state.yt.stopVideo();
        } catch (error) {}
      }
      setPlaybackProvider("spotify");
      setStatus("");
      postPlayerDebug("spotify_fallback_ok", {
        code: Number(reasonCode) || null,
        track: {
          title: state.current.title,
          artist: state.current.artist,
          videoId: state.current.videoId || "",
        },
        note: uri,
      });
      return true;
    } catch (error) {
      postPlayerDebug("spotify_fallback_fail", {
        code: Number(reasonCode) || null,
        track: state.current
          ? { title: state.current.title, artist: state.current.artist, videoId: state.current.videoId || "" }
          : null,
        note: String((error && error.message) || error || ""),
      });
      return false;
    } finally {
      state.spotify.fallbackInFlight = false;
    }
  }

  function coverResolveKeyFromTrack(track) {
    if (!track) return "";
    const trackId = String(track.trackId || track.videoId || "").trim();
    if (trackId) return `id:${trackId}`;
    const title = normalizeKeyPart(track.title || "");
    const artist = normalizeKeyPart(track.artist || "");
    if (!title && !artist) return "";
    return `q:${artist}|${title}`;
  }

  function stampImageTrackMeta(img, track) {
    if (!img) return;
    if (!track) {
      delete img.dataset.trackId;
      delete img.dataset.videoId;
      delete img.dataset.trackTitle;
      delete img.dataset.trackArtist;
      return;
    }
    img.dataset.trackId = String(track.trackId || track.videoId || "");
    img.dataset.videoId = String(track.videoId || "");
    img.dataset.trackTitle = String(track.title || "");
    img.dataset.trackArtist = String(track.artist || "");
  }

  function readImageTrackMeta(img) {
    if (!img) return null;
    const trackId = String(img.dataset.trackId || "").trim();
    const title = String(img.dataset.trackTitle || "").trim();
    const artist = String(img.dataset.trackArtist || "").trim();
    if (!trackId && !title && !artist) return null;
    return {
      trackId,
      title,
      artist,
      videoId: String(img.dataset.videoId || "").trim(),
    };
  }

  async function resolveCoverCandidatesWithYtDlp(track) {
    const key = coverResolveKeyFromTrack(track);
    if (!key) return null;
    if (state.coverResolveCache.has(key)) {
      return state.coverResolveCache.get(key);
    }
    if (state.coverResolvePending.has(key)) {
      return state.coverResolvePending.get(key);
    }

    const pending = (async function () {
      try {
        const params = new URLSearchParams({
          track_id: track.trackId || track.videoId || "",
          video_id: track.videoId || "",
          title: track.title || "",
          artist: track.artist || "",
          covers_only: "1",
        });
        const data = await requestJSON(
          `/api/resolve_video?${params.toString()}`,
          {},
          { timeoutMs: 15000, retries: 1 },
        );
        const resolvedCandidates = prioritizeCoverCandidates([
          ...(Array.isArray(data.thumbnail_candidates)
            ? data.thumbnail_candidates
            : []),
          data.thumbnail,
        ]);
        const payload =
          resolvedCandidates.length || isYoutubeId(data.video_id)
            ? {
                videoId: isYoutubeId(data.video_id) ? String(data.video_id) : "",
                coverCandidates: resolvedCandidates,
              }
            : null;
        state.coverResolveCache.set(key, payload);
        return payload;
      } catch (error) {
        return null;
      } finally {
        state.coverResolvePending.delete(key);
      }
    })();

    state.coverResolvePending.set(key, pending);
    return pending;
  }

  async function maybeRecoverImageCover(img, fallback, options) {
    if (!img) return;
    const opts = options || {};
    if (!opts.allowYtDlpResolve) return;
    if (img.dataset.coverResolveAttempted === "1") return;

    const track = opts.track || readImageTrackMeta(img);
    if (!track) return;

    img.dataset.coverResolveAttempted = "1";
    const token = String(img.dataset.coverLoadToken || "");
    const resolved = await resolveCoverCandidatesWithYtDlp(track);
    if (!resolved || !Array.isArray(resolved.coverCandidates) || !resolved.coverCandidates.length) {
      return;
    }
    if (String(img.dataset.coverLoadToken || "") !== token) return;

    if (opts.track) {
      const merged = prioritizeCoverCandidates([
        ...resolved.coverCandidates,
        ...(Array.isArray(opts.track.coverCandidates) ? opts.track.coverCandidates : []),
      ]);
      opts.track.coverCandidates = merged;
    }

    setImgWithFallback(
      img,
      prioritizeCoverCandidates([
        ...resolved.coverCandidates,
        ...(Array.isArray(opts.track && opts.track.coverCandidates)
          ? opts.track.coverCandidates
          : []),
      ]),
      fallback,
      {
        ...opts,
        allowYtDlpResolve: false,
      },
    );
  }

  function setImgWithFallback(img, candidates, fallback, options) {
    if (!img) return;
    const opts = options || {};
    if (opts.track) {
      stampImageTrackMeta(img, opts.track);
    } else if (!readImageTrackMeta(img)) {
      stampImageTrackMeta(img, null);
    }
    img.dataset.coverResolveAttempted = "0";
    const nextToken = String((Number(img.dataset.coverLoadToken || "0") || 0) + 1);
    img.dataset.coverLoadToken = nextToken;
    const queue = uniq([...(candidates || []), fallback]);
    let i = 0;
    const loadNext = function () {
      if (i >= queue.length) {
        img.onerror = null;
        return;
      }
      const nextUrl = queue[i];
      const isYtThumb = isYoutubeThumbUrl(nextUrl);
      const isFallback = isLocalFallbackCoverUrl(nextUrl);
      img.classList.toggle("yt-cover-thumb", isYtThumb);
      img.classList.toggle("cover-is-fallback", isFallback);

      const shell = img.parentElement;
      if (shell && shell.classList.contains("cover-bg-shell")) {
        shell.classList.toggle("yt-cover-shell", isYtThumb);
        shell.classList.toggle("fallback-cover-shell", isFallback);
        if (isYtThumb) {
          shell.style.setProperty("--cover-bg-image", cssUrlValue(nextUrl));
        } else {
          shell.style.removeProperty("--cover-bg-image");
        }
      }

      const card = img.closest(".avatar-card");
      if (card) {
        card.classList.toggle("avatar-card--fallback", isFallback);
      }

      img.src = nextUrl;
      const shouldUpgradeWeakCover =
        isYtThumb &&
        Boolean(opts.allowYtDlpResolve) &&
        Boolean(opts.track) &&
        !trackHasNonYoutubeCover(opts.track);
      if (isFallback || shouldUpgradeWeakCover) {
        void maybeRecoverImageCover(img, fallback, opts);
      }
      i += 1;
    };
    img.onerror = loadNext;
    loadNext();
  }

  function mapTrack(raw) {
    const rawTrackId = String(raw.id || raw.video_id || "").trim();
    const videoId = chooseYoutubeId([raw.video_id]);
    const trackId = rawTrackId || videoId;
    const backend = Array.isArray(raw.cover_candidates)
      ? raw.cover_candidates.slice(0, 8)
      : [];

    return {
      trackId,
      videoId,
      title: stripFeatureSuffix(raw.title_short || raw.title || "Unknown"),
      artist: stripFeatureSuffix(
        (raw.artist && raw.artist.name) || raw.artist || "Unknown",
      ),
      description: shortDescription(raw.description),
      instrumental: toOptionalBool(raw.instrumental),
      instrumentalConfidence: Number(raw.instrumental_confidence || 0),
      coverCandidates: prioritizeCoverCandidates([
        raw.cover_url,
        raw.cover,
        raw.album && raw.album.cover_medium,
        raw.album && raw.album.cover_big,
        raw.album && raw.album.cover_xl,
        ...backend,
        youtubeThumb(videoId, "maxresdefault"),
        youtubeThumb(videoId, "hq720"),
        youtubeThumb(videoId, "sddefault"),
        youtubeThumb(videoId, "hqdefault"),
        youtubeThumb(videoId, "mqdefault"),
        youtubeThumb(videoId, "default"),
        raw.thumbnail,
      ]),
    };
  }

  function mapRecommendation(raw) {
    const backend = Array.isArray(raw.cover_candidates)
      ? raw.cover_candidates.slice(0, 12)
      : Array.isArray(raw.thumbnail_candidates)
        ? raw.thumbnail_candidates.slice(0, 12)
        : [];
    const rawTrackId = String(
      raw.track_id || raw.id || raw.video_id || "",
    ).trim();
    // Recommendation `track_id` / `id` are dataset IDs, not guaranteed YouTube IDs.
    // Trust only explicit backend `video_id` here; resolve on click if missing.
    const videoId = chooseYoutubeId([raw.video_id]);
    const trackId = rawTrackId || videoId;

    return {
      trackId,
      videoId,
      title: stripFeatureSuffix(raw.title || "Unknown"),
      artist: stripFeatureSuffix(raw.artist || raw.artist_name || "Unknown"),
      description: shortDescription(raw.description),
      instrumental: toOptionalBool(raw.instrumental),
      instrumentalConfidence: Number(raw.instrumental_confidence || 0),
      coverCandidates: prioritizeCoverCandidates([
        raw.cover_url,
        raw.cover,
        raw.album_cover,
        raw.thumbnail,
        ...backend,
        youtubeThumb(videoId, "maxresdefault"),
        youtubeThumb(videoId, "hq720"),
        youtubeThumb(videoId, "sddefault"),
        youtubeThumb(videoId, "hqdefault"),
        youtubeThumb(videoId, "mqdefault"),
        youtubeThumb(videoId, "default"),
      ]),
    };
  }

  function renderEmpty(container, text) {
    if (!container) return;
    container.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = text;
    container.appendChild(empty);
  }

  function renderForYouSkeleton(count) {
    if (!$.forYouGrid) return;
    $.forYouGrid.innerHTML = "";
    const frag = document.createDocumentFragment();
    for (let i = 0; i < count; i += 1) {
      const card = document.createElement("div");
      card.className = "skeleton-card";
      const shimmer = document.createElement("div");
      shimmer.className = "skeleton-shimmer";
      card.appendChild(shimmer);
      frag.appendChild(card);
    }
    $.forYouGrid.appendChild(frag);
    updateForYouStripControls();
  }

  function updatePersonalSummary() {
    if (!$.forYouSummary) return;
    const interactions = Number(state.profile.interaction_count || 0);
    const likes = state.likedSongIds.size;
    const dislikes = state.dislikedSongIds.size;
    const playlistTracks = Number(state.profile.playlist_track_count || 0);
    const identity = state.auth.authenticated
      ? (state.auth.username || `User ${String(state.userId || "").slice(0, 8)}`)
      : (state.userId ? `User ${state.userId.slice(0, 8)}` : "Guest");
    $.forYouSummary.textContent = `${identity} | interactions ${interactions} | likes ${likes} | dislikes ${dislikes} | playlist tracks ${playlistTracks}`;
  }

  function updateSelectedCard() {
    document
      .querySelectorAll(".card, .avatar-card, .list-item")
      .forEach(function (card) {
        const isSelected =
          state.current &&
          card.dataset.index === String(state.queueIndex) &&
          card.dataset.collection === state.activeListKey;
        card.classList.toggle("active", Boolean(isSelected));
      });
  }

  function syncLikeUI() {
    const likeButtons = document.querySelectorAll(".card-like, #like-btn, .list-like-btn");
    likeButtons.forEach(function (btn) {
      const songId = String(btn.dataset.songId || "");
      const liked = songId && state.likedSongIds.has(songId);
      const pending = songId && state.pendingLikeSongIds.has(songId);

      if (btn.classList.contains("list-like-btn")) {
        btn.innerHTML = liked
          ? '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M2 13.5C2 18 5.5 21 10 21c2.7 0 4.3-1.2 5.3-2.4.8-.9 1.2-1.8 1.4-2.4h2.8c.8 0 1.4-.6 1.5-1.4l.8-7.3A1.5 1.5 0 0020.3 6H14V4.8c0-1.8.3-3.2-1.2-3.7-1.4-.5-2.3.9-2.8 2L8.2 6H4.5A2.5 2.5 0 002 8.5v5z"/></svg>'
          : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 9V5.8c0-1.4-.8-2.8-2.1-3.4l-3 6.6H5a2 2 0 00-2 2v6a2 2 0 002 2h8.5a4 4 0 003.9-3.2l1-5.2A1.6 1.6 0 0016.9 9H14z"/></svg>';
        btn.classList.toggle("is-active", Boolean(liked));
      } else {
        btn.classList.toggle("liked", liked);
      }

      btn.disabled = !songId || pending;
      btn.setAttribute("aria-pressed", liked ? "true" : "false");
    });

    const dislikeButtons = document.querySelectorAll(".list-dislike-btn");
    dislikeButtons.forEach(function (btn) {
      const songId = String(btn.dataset.songId || "");
      const disliked = songId && state.dislikedSongIds.has(songId);
      const pending = songId && state.pendingDislikeSongIds.has(songId);
      btn.classList.toggle("is-active", Boolean(disliked));
      btn.disabled = !songId || pending;
      btn.setAttribute("aria-pressed", disliked ? "true" : "false");
    });
  }

  function createListCard(track, index, listKey) {
    const card = document.createElement("article");
    card.className = "list-item";
    card.tabIndex = 0;
    card.dataset.index = String(index);
    card.dataset.collection = listKey;

    const left = document.createElement("div");
    left.className = "list-item-left";

    const coverShell = document.createElement("div");
    coverShell.className = "list-cover-shell cover-bg-shell";

    const cover = document.createElement("img");
    cover.className = "list-cover";
    cover.alt = `${track.title} cover`;
    cover.loading = "lazy";
    cover.decoding = "async";
    setImgWithFallback(cover, track.coverCandidates, CARD_PLACEHOLDER, {
      track,
      allowYtDlpResolve: true,
    });

    const meta = document.createElement("div");
    meta.className = "list-meta";

    const title = document.createElement("div");
    title.className = "list-title";
    title.textContent = track.title;

    const artist = document.createElement("div");
    artist.className = "list-artist";
    artist.textContent = track.artist;

    meta.appendChild(title);
    meta.appendChild(artist);
    coverShell.appendChild(cover);
    left.appendChild(coverShell);
    left.appendChild(meta);

    const right = document.createElement("div");
    right.className = "list-right";

    const songId = songIdForTrack(track);
    const playlistBtn = document.createElement("button");
    playlistBtn.type = "button";
    playlistBtn.className = "list-action-btn list-playlist-btn";
    playlistBtn.setAttribute("aria-label", "Add to playlist");
    playlistBtn.title = "Add to Playlist";
    playlistBtn.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M5 7h10M5 12h10M5 17h6M18 14v6M15 17h6"/></svg>';
    playlistBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      void openPlaylistPicker(track);
    });

    const dislikeBtn = document.createElement("button");
    dislikeBtn.type = "button";
    dislikeBtn.className = "list-action-btn list-dislike-btn";
    dislikeBtn.setAttribute("aria-label", "Thumbs down");
    dislikeBtn.title = "Dislike";
    dislikeBtn.dataset.songId = songId;
    dislikeBtn.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M10 15v3.2c0 1.4.8 2.8 2.1 3.4l3-6.6H19a2 2 0 002-2v-6a2 2 0 00-2-2h-8.5a4 4 0 00-3.9 3.2l-1 5.2A1.6 1.6 0 007.1 15H10z"/></svg>';
    dislikeBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      if (!songId) return;
      void toggleDislikeSong(songId);
    });

    const likeBtn = document.createElement("button");
    likeBtn.type = "button";
    likeBtn.className = "list-action-btn list-like-btn";
    likeBtn.style.color = "var(--text-secondary-dark)";
    likeBtn.setAttribute("aria-label", "Thumbs up");
    likeBtn.title = "Like";
    likeBtn.dataset.songId = songId;
    likeBtn.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 9V5.8c0-1.4-.8-2.8-2.1-3.4l-3 6.6H5a2 2 0 00-2 2v6a2 2 0 002 2h8.5a4 4 0 003.9-3.2l1-5.2A1.6 1.6 0 0016.9 9H14z"/></svg>';
    likeBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      if (!songId) return;
      void toggleLikeSong(songId);
    });

    right.appendChild(playlistBtn);
    right.appendChild(dislikeBtn);
    right.appendChild(likeBtn);
    card.appendChild(left);
    card.appendChild(right);

    card.addEventListener("click", function () {
      void playTrackAt(index, listKey);
    });

    return card;
  }

  function createAvatarCard(track, index, listKey) {
    const card = document.createElement("article");
    card.className = "avatar-card";
    card.tabIndex = 0;
    card.dataset.index = String(index);
    card.dataset.collection = listKey;

    const coverShell = document.createElement("div");
    coverShell.className = "avatar-cover-shell cover-bg-shell";

    const cover = document.createElement("img");
    cover.className = "avatar-cover";
    cover.alt = `${track.title} cover`;
    cover.loading = "lazy";
    cover.decoding = "async";
    setImgWithFallback(cover, track.coverCandidates, CARD_PLACEHOLDER, {
      track,
      allowYtDlpResolve: true,
    });

    const title = document.createElement("div");
    title.className = "avatar-title";
    title.textContent = track.title;
    title.title = track.title;

    coverShell.appendChild(cover);
    card.appendChild(coverShell);
    card.appendChild(title);

    card.addEventListener("click", function () {
      void playTrackAt(index, listKey);
    });

    return card;
  }

  function renderSearchResults() {
    if (!$.results) return;
    $.results.innerHTML = "";
    const tracks = state.lists.search;
    if (!tracks.length) {
      renderEmpty($.results, "Could not find close matches right now. Try another prompt.");
      return;
    }

    const frag = document.createDocumentFragment();
    tracks.forEach(function (track, index) {
      frag.appendChild(createListCard(track, index, "search"));
    });
    $.results.appendChild(frag);
    updateSelectedCard();
    syncLikeUI();
    queueLyricsPrefetchForVisibleTracks(
      state.lists.search,
      LYRICS_PREFETCH_TRACK_LIMIT_SEARCH,
    );
  }

  function getHomeQueueTotalPages() {
    const pageSize = Math.max(1, Number(state.homePaging.pageSize) || HOME_QUEUE_BATCH_SIZE);
    const tracks = Array.isArray(state.lists.home) ? state.lists.home : [];
    if (!tracks.length) return 0;
    return Math.max(1, Math.ceil(tracks.length / pageSize));
  }

  function clampHomeQueueCurrentPage() {
    const totalPages = getHomeQueueTotalPages();
    if (!totalPages) {
      state.homePaging.currentPage = 0;
      return 0;
    }
    const next = Math.max(
      0,
      Math.min(
        Number(state.homePaging.currentPage) || 0,
        totalPages - 1,
      ),
    );
    state.homePaging.currentPage = next;
    return next;
  }

  function getHomeQueuePageSlice() {
    const pageSize = Math.max(1, Number(state.homePaging.pageSize) || HOME_QUEUE_BATCH_SIZE);
    const tracks = Array.isArray(state.lists.home) ? state.lists.home : [];
    const currentPage = clampHomeQueueCurrentPage();
    const start = currentPage * pageSize;
    const end = start + pageSize;
    return {
      start,
      tracks: tracks.slice(start, end),
      total: tracks.length,
      totalPages: getHomeQueueTotalPages(),
    };
  }

  function updateHomeQueuePagerUI() {
    if (!$.homeQueuePager || !$.homeQueuePrevBtn || !$.homeQueueNextBtn) return;
    const totalPages = getHomeQueueTotalPages();
    const hasPages = totalPages > 0;
    const currentPage = clampHomeQueueCurrentPage();
    const source = state.homeQueueSource || {};
    const isLastPage = hasPages && currentPage >= totalPages - 1;
    $.homeQueuePager.classList.toggle("hidden", !hasPages);
    if ($.homeQueuePageLabel) {
      $.homeQueuePageLabel.textContent = hasPages
        ? `Page ${currentPage + 1} of ${totalPages}`
        : "";
    }
    $.homeQueuePrevBtn.disabled = !hasPages || currentPage <= 0 || Boolean(source.loading);
    $.homeQueueNextBtn.textContent = `Next ${Math.max(1, Number(state.homePaging.pageSize) || HOME_QUEUE_BATCH_SIZE)}`;
    $.homeQueueNextBtn.disabled =
      !hasPages ||
      Boolean(source.loading) ||
      (isLastPage && Boolean(source.exhausted));
  }

  async function goToHomeQueuePage(pageIndex) {
    const requestedPage = Number(pageIndex);
    if (!Number.isInteger(requestedPage) || requestedPage < 0) return;
    let totalPages = getHomeQueueTotalPages();
    const source = state.homeQueueSource || {};
    const wantsForward = requestedPage > Number(state.homePaging.currentPage || 0);

    if (wantsForward && requestedPage >= totalPages && !source.exhausted) {
      setHomeStatus("Loading more queue tracks...");
      let attempts = 0;
      while (requestedPage >= totalPages && !source.exhausted && attempts < HOME_QUEUE_AUTOFETCH_MAX_PAGES) {
        const appended = await appendMoreHomeQueueFromSource(false);
        attempts += 1;
        totalPages = getHomeQueueTotalPages();
        if (!appended) break;
      }
      if (requestedPage >= totalPages && source.exhausted) {
        setHomeStatus("No more tracks for this Home query.");
      } else {
        setHomeStatus("");
      }
    }

    totalPages = getHomeQueueTotalPages();
    if (!totalPages) {
      renderHomeQueue();
      return;
    }
    state.homePaging.currentPage = Math.max(0, Math.min(requestedPage, totalPages - 1));
    renderHomeQueue();
  }

  function renderHomeQueue() {
    if (!$.homeQueueResults) return;
    $.homeQueueResults.innerHTML = "";
    const tracks = Array.isArray(state.lists.home) ? state.lists.home : [];
    if ($.homeView) {
      $.homeView.classList.toggle("home-empty", !tracks.length);
    }
    if (!tracks.length) {
      renderEmpty($.homeQueueResults, "No queued songs yet. Use the Home search box to start playback.");
      updateHomeQueuePagerUI();
      return;
    }

    const page = getHomeQueuePageSlice();
    const frag = document.createDocumentFragment();
    page.tracks.forEach(function (track, index) {
      frag.appendChild(createListCard(track, page.start + index, "home"));
    });
    $.homeQueueResults.appendChild(frag);
    updateSelectedCard();
    syncLikeUI();
    updateHomeQueuePagerUI();
  }

  function renderLibraryList(container, tracks, listKey, emptyText) {
    if (!container) return;
    container.innerHTML = "";
    const rows = Array.isArray(tracks) ? tracks : [];
    if (!rows.length) {
      renderEmpty(container, emptyText);
      return;
    }
    const frag = document.createDocumentFragment();
    rows.forEach(function (track, index) {
      frag.appendChild(createListCard(track, index, listKey));
    });
    container.appendChild(frag);
    updateSelectedCard();
    syncLikeUI();
  }

  function renderLikedSongs() {
    renderLibraryList(
      $.likedResults,
      state.lists.liked,
      "liked",
      "No liked songs yet. Tap the thumbs-up icon on any track to save it here.",
    );
  }

  function renderHistoryTracks() {
    renderLibraryList(
      $.historyResults,
      state.lists.history,
      "history",
      "No listening history yet. Play a song and it will appear here.",
    );
  }

  function playlistSelectionKey() {
    const uid = String(state.userId || "guest").trim() || "guest";
    return `${PLAYLISTS_SELECTED_KEY_PREFIX}:${uid}`;
  }

  function rememberSelectedPlaylistId(playlistId) {
    const id = String(playlistId || "").trim();
    if (!id) return;
    try {
      localStorage.setItem(playlistSelectionKey(), id);
    } catch (error) {}
  }

  function restoreSelectedPlaylistId() {
    try {
      return String(localStorage.getItem(playlistSelectionKey()) || "").trim();
    } catch (error) {
      return "";
    }
  }

  function normalizePlaylistFromApi(raw) {
    if (!raw || typeof raw !== "object") return null;
    const id = String(raw.playlist_id || raw.id || "").trim();
    if (!id) return null;
    const name = String(raw.name || "").trim() || "Untitled Playlist";
    const tracks = mapProfileRows(raw.tracks);
    return {
      id,
      name,
      tracks,
      coverImage: String(raw.cover_image || "").trim(),
      createdAt: Number(raw.created_at || 0),
      updatedAt: Number(raw.updated_at || 0),
      trackCount: Number(raw.track_count || tracks.length || 0),
    };
  }

  function playlistTrackCount(playlist) {
    if (!playlist || typeof playlist !== "object") return 0;
    if (playlist.trackCount != null) return Math.max(0, Number(playlist.trackCount || 0));
    return Array.isArray(playlist.tracks) ? playlist.tracks.length : 0;
  }

  function playlistCoverCandidates(playlist) {
    const out = [];
    if (playlist && playlist.coverImage) out.push(playlist.coverImage);
    const tracks = Array.isArray(playlist && playlist.tracks) ? playlist.tracks : [];
    for (let i = 0; i < tracks.length && out.length < 6; i += 1) {
      const track = tracks[i];
      if (!track) continue;
      if (Array.isArray(track.coverCandidates)) {
        track.coverCandidates.forEach(function (url) {
          if (url) out.push(url);
        });
      }
    }
    const deduped = [];
    const seen = new Set();
    out.forEach(function (url) {
      const key = String(url || "").trim();
      if (!key || seen.has(key)) return;
      seen.add(key);
      deduped.push(key);
    });
    return deduped;
  }

  function getSelectedPlaylist() {
    const selectedId = String(state.playlists.selectedId || "").trim();
    let selected = (state.playlists.items || []).find(function (item) {
      return item.id === selectedId;
    });
    if (!selected && (state.playlists.items || []).length) {
      selected = state.playlists.items[0];
      state.playlists.selectedId = selected.id;
      rememberSelectedPlaylistId(selected.id);
    }
    return selected || null;
  }

  function renderPlaylistTabs() {
    if (!$.playlistTabs) return;
    $.playlistTabs.innerHTML = "";
    const frag = document.createDocumentFragment();
    (state.playlists.items || []).forEach(function (playlist) {
      const card = document.createElement("div");
      card.className = "playlist-tab playlist-card";
      const isActive = playlist.id === state.playlists.selectedId;
      card.classList.toggle("active", isActive);
      card.setAttribute("role", "tab");
      card.setAttribute("aria-selected", isActive ? "true" : "false");
      card.dataset.playlistId = playlist.id;

      const selectBtn = document.createElement("button");
      selectBtn.type = "button";
      selectBtn.className = "playlist-card-hit";
      selectBtn.dataset.playlistId = playlist.id;
      selectBtn.setAttribute("aria-label", `Open playlist ${playlist.name}`);

      const coverShell = document.createElement("div");
      coverShell.className = "playlist-card-cover-shell cover-bg-shell";

      const cover = document.createElement("img");
      cover.className = "playlist-card-cover";
      cover.alt = `${playlist.name} playlist cover`;
      cover.loading = "lazy";
      cover.decoding = "async";
      setImgWithFallback(cover, playlistCoverCandidates(playlist), CARD_PLACEHOLDER, {});

      const meta = document.createElement("div");
      meta.className = "playlist-card-meta";

      const name = document.createElement("div");
      name.className = "playlist-card-name";
      name.textContent = playlist.name;

      const metaLine = document.createElement("div");
      metaLine.className = "playlist-card-sub";
      const trackCount = playlistTrackCount(playlist);
      metaLine.textContent = `By You • ${trackCount} track${trackCount === 1 ? "" : "s"}`;

      const coverBtn = document.createElement("button");
      coverBtn.type = "button";
      coverBtn.className = "playlist-card-cover-btn";
      coverBtn.dataset.playlistCoverFor = playlist.id;
      coverBtn.setAttribute("aria-label", `Edit cover for ${playlist.name}`);
      coverBtn.title = "Edit Cover";
      coverBtn.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7h4l2-2h4l2 2h4v10H4z"/><circle cx="12" cy="12" r="3"/></svg>';

      const renameBtn = document.createElement("button");
      renameBtn.type = "button";
      renameBtn.className = "playlist-card-rename-btn";
      renameBtn.dataset.playlistRenameFor = playlist.id;
      renameBtn.setAttribute("aria-label", `Rename playlist ${playlist.name}`);
      renameBtn.title = "Rename Playlist";
      renameBtn.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 20h4l10-10a2 2 0 0 0-4-4L4 16v4z"/></svg>';

      coverShell.appendChild(cover);
      meta.appendChild(name);
      meta.appendChild(metaLine);
      selectBtn.appendChild(coverShell);
      selectBtn.appendChild(meta);
      card.appendChild(selectBtn);
      card.appendChild(renameBtn);
      card.appendChild(coverBtn);
      frag.appendChild(card);
    });
    $.playlistTabs.appendChild(frag);
  }

  function renderPlaylists() {
    const selected = getSelectedPlaylist();
    const tracks = selected && Array.isArray(selected.tracks) ? selected.tracks : [];
    state.lists.playlists = tracks.slice();
    renderPlaylistTabs();
    updatePlaylistActionButtons(selected);
    if ($.playlistsSummary) {
      if (selected) {
        const count = tracks.length;
        $.playlistsSummary.textContent = `${selected.name} • ${count} track${count === 1 ? "" : "s"}`;
      } else {
        $.playlistsSummary.textContent = "Create a playlist and start adding songs from any list.";
      }
    }
    renderLibraryList(
      $.playlistsResults,
      state.lists.playlists,
      "playlists",
      "No songs in this playlist yet. Use the + button on any track to save it here.",
    );
  }

  function updatePlaylistActionButtons(selectedPlaylist) {
    const hasSelected = Boolean(selectedPlaylist && selectedPlaylist.id);
    if ($.playlistRenameBtn) {
      $.playlistRenameBtn.disabled = !hasSelected;
      $.playlistRenameBtn.title = hasSelected
        ? `Rename ${selectedPlaylist.name}`
        : "Select a playlist first";
    }
    if ($.playlistEditCoverBtn) {
      $.playlistEditCoverBtn.disabled = !hasSelected;
      $.playlistEditCoverBtn.title = hasSelected
        ? `Edit cover for ${selectedPlaylist.name}`
        : "Select a playlist first";
    }
    if ($.playlistClearBtn) {
      $.playlistClearBtn.disabled = !hasSelected;
      $.playlistClearBtn.title = hasSelected
        ? `Clear tracks from ${selectedPlaylist.name}`
        : "Select a playlist first";
    }
    if ($.playlistDeleteBtn) {
      $.playlistDeleteBtn.disabled = !hasSelected;
      $.playlistDeleteBtn.title = hasSelected
        ? `Delete ${selectedPlaylist.name}`
        : "Select a playlist first";
    }
  }

  function setPlaylistsFromPayload(payload) {
    const items = (Array.isArray(payload && payload.playlists) ? payload.playlists : [])
      .map(normalizePlaylistFromApi)
      .filter(Boolean);
    const restoredSelectedId = restoreSelectedPlaylistId();
    const selectedId = String(state.playlists.selectedId || restoredSelectedId || "").trim();
    state.playlists.items = items;
    if (selectedId && items.some(function (it) { return it.id === selectedId; })) {
      state.playlists.selectedId = selectedId;
    } else if (items.length) {
      state.playlists.selectedId = items[0].id;
    } else {
      state.playlists.selectedId = "";
    }
    if (state.playlists.selectedId) {
      rememberSelectedPlaylistId(state.playlists.selectedId);
    }
    state.profile.playlist_count = items.length;
    state.profile.playlist_track_count = items.reduce(function (sum, playlist) {
      const count = Array.isArray(playlist.tracks)
        ? playlist.tracks.length
        : Number(playlist.trackCount || 0);
      return sum + Math.max(0, Number(count || 0));
    }, 0);
    updatePersonalSummary();
    state.playlistsLoaded = true;
  }

  function friendlyPlaylistApiError(error, fallbackText) {
    const fallback = String(fallbackText || "Playlist request failed.");
    const status = Number(error && error.status ? error.status : 0);
    const raw = String((error && error.message) || "").trim();
    const looksHtml = raw.startsWith("<!doctype") || raw.startsWith("<html");
    if (status === 404 || looksHtml) {
      return "Playlist API not found (404). Restart the Flask backend so the new playlist routes load.";
    }
    if (!raw) return fallback;
    return raw.length > 180 ? `${raw.slice(0, 177)}...` : raw;
  }

  async function fetchPlaylistsFromApi(options) {
    if (!state.userId) return { playlists: [] };
    const opts = options || {};
    const covers = opts.covers ? "1" : "0";
    return requestJSON(
      `/api/playlists?user_id=${encodeURIComponent(state.userId)}&tracks=1&track_limit=${PROFILE_LIBRARY_FETCH_SIZE}&covers=${covers}`,
      {},
      { timeoutMs: 10000, retries: 0 },
    );
  }

  async function loadPlaylists(isSilent) {
    if (!state.userId) return;
    if (!isSilent) setPlaylistsStatus("Loading playlists...");
    try {
      const payload = await fetchPlaylistsFromApi({ covers: false });
      setPlaylistsFromPayload(payload || {});

      // Auto-create a default playlist for first-time users so add-to-playlist works immediately.
      if (!(state.playlists.items || []).length) {
        await requestJSON(
          "/api/playlists/create",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              user_id: state.userId,
              name: DEFAULT_PLAYLIST_NAME,
            }),
          },
          { timeoutMs: 8000, retries: 0 },
        );
        const refreshed = await fetchPlaylistsFromApi({ covers: false });
        setPlaylistsFromPayload(refreshed || {});
      }

      renderPlaylists();
      if (!isSilent) setPlaylistsStatus("");
      return true;
    } catch (error) {
      state.playlistsLoaded = true;
      state.playlists.items = [];
      state.playlists.selectedId = "";
      renderPlaylists();
      const msg = friendlyPlaylistApiError(error, "Could not load playlists.");
      if (!isSilent) setPlaylistsStatus(msg, true);
      if (state.playlistPicker && state.playlistPicker.open) {
        setPlaylistModalStatus(msg, true);
      }
      return false;
    }
  }

  async function createPlaylist(nameOverride) {
    if (!state.userId) return null;
    const rawName = nameOverride != null ? String(nameOverride || "") : "";
    const name = String(rawName || "").trim();
    if (!name) {
      setPlaylistsStatus("Playlist name cannot be empty.", true);
      setPlaylistModalStatus("Playlist name cannot be empty.", true);
      return null;
    }
    try {
      const res = await requestJSON(
        "/api/playlists/create",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            name,
          }),
        },
        { timeoutMs: 8000, retries: 0 },
      );
      const playlistId = String(res && res.playlist_id ? res.playlist_id : "").trim();
      await loadPlaylists(true);
      if (playlistId) {
        state.playlists.selectedId = playlistId;
        rememberSelectedPlaylistId(playlistId);
        renderPlaylists();
      }
      setPlaylistsStatus(`Created playlist "${name}".`);
      return playlistId || (getSelectedPlaylist() && getSelectedPlaylist().id) || null;
    } catch (error) {
      const msg = friendlyPlaylistApiError(error, "Could not create playlist.");
      setPlaylistsStatus(msg, true);
      setPlaylistModalStatus(msg, true);
      return null;
    }
  }

  async function renamePlaylist(playlistId, nameOverride) {
    if (!state.userId) return false;
    const id = String(playlistId || "").trim();
    const name = String(nameOverride || "").trim();
    if (!id) {
      setPlaylistsStatus("Playlist not found.", true);
      setPlaylistModalStatus("Playlist not found.", true);
      return false;
    }
    if (!name) {
      setPlaylistsStatus("Playlist name cannot be empty.", true);
      setPlaylistModalStatus("Playlist name cannot be empty.", true);
      return false;
    }
    try {
      await requestJSON(
        "/api/playlists/rename",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            playlist_id: id,
            name,
          }),
        },
        { timeoutMs: 8000, retries: 0 },
      );
      await loadPlaylists(true);
      const selected = playlistById(id);
      if (selected) {
        state.playlists.selectedId = id;
        rememberSelectedPlaylistId(id);
      }
      renderPlaylists();
      setPlaylistsStatus(`Renamed playlist to "${name}".`);
      return true;
    } catch (error) {
      const msg = friendlyPlaylistApiError(error, "Could not rename playlist.");
      setPlaylistsStatus(msg, true);
      setPlaylistModalStatus(msg, true);
      return false;
    }
  }

  async function addTrackToPlaylist(playlistId, track) {
    if (!state.userId) return { ok: false, error: "Missing user." };
    const id = String(playlistId || "").trim();
    const songId = String(songIdForTrack(track) || "").trim();
    if (!id || !songId) {
      return { ok: false, error: "Track or playlist unavailable." };
    }
    try {
      const res = await requestJSON(
        "/api/playlists/add_track",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            playlist_id: id,
            song_id: songId,
          }),
        },
        { timeoutMs: 8000, retries: 0 },
      );
      const wasAdded = Boolean(
        res && (res.added === true || String(res.status || "") === "added"),
      );
      await loadPlaylists(true);
      const target = playlistById(id);
      const targetName = target ? target.name : "playlist";
      if (wasAdded) {
        setPlaylistsStatus(`Added to ${targetName}.`);
        scheduleForYouRefresh(650);
        return { ok: true, duplicate: false };
      }
      setPlaylistsStatus(`Already in ${targetName}.`);
      return { ok: true, duplicate: true };
    } catch (error) {
      return {
        ok: false,
        error: friendlyPlaylistApiError(error, "Could not add track to playlist."),
      };
    }
  }

  function openPlaylistCoverPickerFor(playlistId) {
    const id = String(playlistId || "").trim();
    if (!id || !$.playlistCoverInput) return;
    state.playlistCoverUpload.targetPlaylistId = id;
    try {
      $.playlistCoverInput.value = "";
    } catch (error) {}
    $.playlistCoverInput.click();
  }

  function readFileAsDataUrl(file) {
    return new Promise(function (resolve, reject) {
      const reader = new FileReader();
      reader.onload = function () {
        resolve(String(reader.result || ""));
      };
      reader.onerror = function () {
        reject(new Error("Could not read image file."));
      };
      reader.readAsDataURL(file);
    });
  }

  async function uploadPlaylistCoverFromFile(playlistId, file) {
    const id = String(playlistId || "").trim();
    if (!id || !state.userId || !file) return false;
    const mime = String(file.type || "").toLowerCase();
    if (!mime.startsWith("image/")) {
      setPlaylistsStatus("Please choose an image file for the playlist cover.", true);
      return false;
    }
    const maxBytes = 1_500_000; // Keep data URLs reasonably small for SQLite/local dev.
    if (Number(file.size || 0) > maxBytes) {
      setPlaylistsStatus("Cover image is too large. Choose an image under 1.5 MB.", true);
      return false;
    }
    try {
      state.playlistCoverUpload.busy = true;
      setPlaylistsStatus("Uploading playlist cover...");
      const dataUrl = await readFileAsDataUrl(file);
      await requestJSON(
        "/api/playlists/set_cover",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            playlist_id: id,
            cover_image: dataUrl,
          }),
        },
        { timeoutMs: 12000, retries: 0 },
      );
      await loadPlaylists(true);
      setPlaylistsStatus("Playlist cover updated.");
      return true;
    } catch (error) {
      setPlaylistsStatus(
        friendlyPlaylistApiError(error, "Could not update playlist cover."),
        true,
      );
      return false;
    } finally {
      state.playlistCoverUpload.busy = false;
      state.playlistCoverUpload.targetPlaylistId = "";
      if ($.playlistCoverInput) {
        try {
          $.playlistCoverInput.value = "";
        } catch (innerError) {}
      }
    }
  }

  async function clearSelectedPlaylist() {
    const selected = getSelectedPlaylist();
    if (!selected) return;
    if (!(selected.tracks || []).length) {
      setPlaylistsStatus("Selected playlist is already empty.");
      return;
    }
    const ok = await openConfirmDialog({
      title: "Clear Playlist",
      message: `Clear all tracks from "${selected.name}"?`,
      confirmText: "Clear",
      cancelText: "Keep",
      tone: "warning",
    });
    if (!ok) return;
    try {
      await requestJSON(
        "/api/playlists/clear",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            playlist_id: selected.id,
          }),
        },
        { timeoutMs: 8000, retries: 0 },
      );
      await loadPlaylists(true);
      setPlaylistsStatus(`Cleared "${selected.name}".`);
      scheduleForYouRefresh(650);
    } catch (error) {
      const msg = friendlyPlaylistApiError(error, "Could not clear playlist.");
      setPlaylistsStatus(msg, true);
      if (state.playlistPicker && state.playlistPicker.open) {
        setPlaylistModalStatus(msg, true);
      }
    }
  }

  async function deleteSelectedPlaylist() {
    const selected = getSelectedPlaylist();
    if (!selected) return;
    const trackCount = Array.isArray(selected.tracks) ? selected.tracks.length : 0;
    const ok = await openConfirmDialog({
      title: "Delete Playlist",
      message: `Delete "${selected.name}" permanently?${trackCount ? ` (${trackCount} track${trackCount === 1 ? "" : "s"})` : ""}`,
      confirmText: "Delete",
      cancelText: "Cancel",
      tone: "danger",
    });
    if (!ok) return;
    try {
      await requestJSON(
        "/api/playlists/delete",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            playlist_id: selected.id,
          }),
        },
        { timeoutMs: 8000, retries: 0 },
      );
      await loadPlaylists(true);
      setPlaylistsStatus(`Deleted "${selected.name}".`);
      scheduleForYouRefresh(650);
    } catch (error) {
      const msg = friendlyPlaylistApiError(error, "Could not delete playlist.");
      setPlaylistsStatus(msg, true);
      if (state.playlistPicker && state.playlistPicker.open) {
        setPlaylistModalStatus(msg, true);
      }
    }
  }

  function selectPlaylist(playlistId) {
    const id = String(playlistId || "").trim();
    if (!id) return;
    const exists = (state.playlists.items || []).some(function (item) {
      return item.id === id;
    });
    if (!exists) return;
    state.playlists.selectedId = id;
    rememberSelectedPlaylistId(id);
    renderPlaylists();
    setPlaylistsStatus("");
  }

  function playlistById(playlistId) {
    const id = String(playlistId || "").trim();
    if (!id) return null;
    return (state.playlists.items || []).find(function (item) {
      return item.id === id;
    }) || null;
  }

  function playlistContainsTrack(playlist, track) {
    const songId = String(songIdForTrack(track) || "").trim();
    if (!songId || !playlist || !Array.isArray(playlist.tracks)) return false;
    return playlist.tracks.some(function (item) {
      return String(songIdForTrack(item) || "").trim() === songId;
    });
  }

  function configurePlaylistModalForCurrentMode(options) {
    const opts = options || {};
    const shouldPrefillName = Boolean(opts.prefillName);
    const mode = state.playlistPicker && state.playlistPicker.mode === "rename" ? "rename" : "add";
    const pendingTrack = state.playlistPicker ? state.playlistPicker.track : null;
    const target = playlistById(state.playlistPicker ? state.playlistPicker.targetPlaylistId : "");

    if ($.playlistModalTitle) {
      $.playlistModalTitle.textContent = mode === "rename" ? "Rename Playlist" : "Add To Playlist";
    }
    if ($.playlistModalCreateBtn) {
      $.playlistModalCreateBtn.textContent = mode === "rename" ? "Save" : "Create";
    }
    if ($.playlistModalName) {
      $.playlistModalName.placeholder = mode === "rename" ? "Rename playlist" : "Create a playlist";
      if (mode === "rename" && shouldPrefillName && target) {
        $.playlistModalName.value = target.name || "";
      } else if (mode !== "rename" && shouldPrefillName) {
        $.playlistModalName.value = "";
      }
    }
    if ($.playlistModalSubtitle) {
      if (mode === "rename") {
        $.playlistModalSubtitle.textContent = target
          ? `Update the playlist name for ${target.name}.`
          : "Choose a playlist and update its name.";
      } else if (pendingTrack) {
        const artist = String(pendingTrack.artist || "").trim();
        $.playlistModalSubtitle.textContent = artist
          ? `${pendingTrack.title} • ${artist}`
          : String(pendingTrack.title || "Choose a playlist");
      } else {
        $.playlistModalSubtitle.textContent =
          "Create a playlist or choose an existing playlist to make it active.";
      }
    }
  }

  function openPlaylistRenameEditor(playlistId) {
    const id = String(playlistId || "").trim();
    if (!id) return;
    void openPlaylistPicker(null, { mode: "rename", playlistId: id });
  }

  function renderPlaylistPickerList() {
    if (!$.playlistModalList) return;
    $.playlistModalList.innerHTML = "";
    const mode = state.playlistPicker && state.playlistPicker.mode === "rename" ? "rename" : "add";
    if (mode === "rename") {
      const target = playlistById(state.playlistPicker && state.playlistPicker.targetPlaylistId);
      const note = document.createElement("div");
      note.className = "playlist-picker-empty";
      note.textContent = target
        ? `Renaming "${target.name}". Save to apply the new name.`
        : "Playlist not found. Close and try again.";
      $.playlistModalList.appendChild(note);
      return;
    }

    const items = Array.isArray(state.playlists.items) ? state.playlists.items : [];
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "playlist-picker-empty";
      empty.textContent = "No playlists yet. Create one above to save this track.";
      $.playlistModalList.appendChild(empty);
      return;
    }
    const frag = document.createDocumentFragment();
    const hasPendingTrack = Boolean(state.playlistPicker && state.playlistPicker.track);
    items.forEach(function (playlist) {
      const row = document.createElement("div");
      row.className = "playlist-picker-item";
      row.setAttribute("role", "listitem");

      const main = document.createElement("div");
      main.className = "playlist-picker-item-main";

      const name = document.createElement("div");
      name.className = "playlist-picker-item-name";
      name.textContent = playlist.name;

      const meta = document.createElement("div");
      meta.className = "playlist-picker-item-meta";
      const count = Array.isArray(playlist.tracks) ? playlist.tracks.length : Number(playlist.trackCount || 0);
      const alreadyInPlaylist = hasPendingTrack
        ? playlistContainsTrack(playlist, state.playlistPicker.track)
        : false;
      meta.textContent = alreadyInPlaylist
        ? `${count} track${count === 1 ? "" : "s"} • already added`
        : `${count} track${count === 1 ? "" : "s"}`;

      const action = document.createElement("button");
      action.type = "button";
      action.className = "playlist-picker-item-action";
      action.textContent = hasPendingTrack
        ? (alreadyInPlaylist ? "Added" : "Add")
        : "Select";
      action.dataset.playlistId = playlist.id;
      if (alreadyInPlaylist) {
        action.disabled = true;
      }

      main.appendChild(name);
      main.appendChild(meta);
      row.appendChild(main);
      row.appendChild(action);
      frag.appendChild(row);
    });
    $.playlistModalList.appendChild(frag);
  }

  function closePlaylistPicker() {
    state.playlistPicker.open = false;
    state.playlistPicker.track = null;
    state.playlistPicker.busy = false;
    state.playlistPicker.mode = "add";
    state.playlistPicker.targetPlaylistId = "";
    configurePlaylistModalForCurrentMode({ prefillName: true });
    setPlaylistModalStatus("");
    if ($.playlistModal) {
      $.playlistModal.classList.remove("open");
      $.playlistModal.setAttribute("aria-hidden", "true");
    }
  }

  async function openPlaylistPicker(track, options) {
    if (!state.userId) return;
    const opts = options || {};
    const mode = opts.mode === "rename" ? "rename" : "add";
    const targetPlaylistId = mode === "rename" ? String(opts.playlistId || "").trim() : "";
    const pickerTrack = mode === "rename" ? null : (track || null);
    state.playlistPicker.open = true;
    state.playlistPicker.track = pickerTrack;
    state.playlistPicker.busy = false;
    state.playlistPicker.mode = mode;
    state.playlistPicker.targetPlaylistId = targetPlaylistId;
    if ($.playlistModal) {
      $.playlistModal.classList.add("open");
      $.playlistModal.setAttribute("aria-hidden", "false");
    }
    configurePlaylistModalForCurrentMode({ prefillName: true });
    if ($.playlistModalName) {
      setTimeout(function () {
        try {
          $.playlistModalName.focus();
        } catch (error) {}
      }, 0);
    }
    setPlaylistModalStatus("Loading playlists...");
    const loaded = await loadPlaylists(true);
    if (mode === "rename") {
      const target = playlistById(targetPlaylistId);
      if (!target) {
        setPlaylistModalStatus("Playlist not found.", true);
        return;
      }
    }
    configurePlaylistModalForCurrentMode({ prefillName: true });
    renderPlaylistPickerList();
    if (loaded) setPlaylistModalStatus("");
  }

  async function createPlaylistFromModal() {
    if (state.playlistPicker.busy) return;
    const name = $.playlistModalName ? $.playlistModalName.value : "";
    const mode = state.playlistPicker && state.playlistPicker.mode === "rename" ? "rename" : "add";
    state.playlistPicker.busy = true;
    try {
      if (mode === "rename") {
        const targetPlaylistId = String(state.playlistPicker.targetPlaylistId || "").trim();
        const renamed = await renamePlaylist(targetPlaylistId, name);
        if (renamed) {
          setPlaylistModalStatus("Playlist renamed.");
          closePlaylistPicker();
        }
        return;
      }

      const playlistId = await createPlaylist(name);
      if ($.playlistModalName) $.playlistModalName.value = "";
      if (playlistId) {
        renderPlaylistPickerList();
        setPlaylistModalStatus("Playlist created. Tap Add to save the track.");
      }
    } finally {
      state.playlistPicker.busy = false;
    }
  }

  async function addPendingTrackToPlaylist(playlistId) {
    const track = state.playlistPicker.track;
    if (state.playlistPicker.busy) return;
    if (!track) {
      selectPlaylist(playlistId);
      const selected = getSelectedPlaylist();
      setPlaylistsStatus(
        selected ? `Selected ${selected.name}.` : "Playlist selected.",
      );
      closePlaylistPicker();
      return;
    }
    state.playlistPicker.busy = true;
    setPlaylistModalStatus("Adding to playlist...");
    try {
      const result = await addTrackToPlaylist(playlistId, track);
      if (!result.ok) {
        setPlaylistModalStatus(result.error || "Could not add track.", true);
        return;
      }
      if (result.duplicate) {
        renderPlaylistPickerList();
        setPlaylistModalStatus("Track is already in that playlist.");
        return;
      }
      renderPlaylistPickerList();
      setPlaylistModalStatus("Added to playlist.");
      closePlaylistPicker();
    } finally {
      state.playlistPicker.busy = false;
    }
  }

  function appendTracksToHomeQueue(tracks, query, options) {
    const opts = options || {};
    const incoming = Array.isArray(tracks) ? tracks : [];
    const replaceExisting = Boolean(opts.replaceExisting);
    const existing = replaceExisting
      ? []
      : Array.isArray(state.lists.home)
        ? state.lists.home
        : [];
    const seen = new Set(
      existing.map(function (track) {
        return (
          songIdForTrack(track) ||
          `${normalizeKeyPart(track && track.title)}-${normalizeKeyPart(track && track.artist)}`
        );
      }),
    );

    const added = [];
    incoming.forEach(function (track) {
      const key =
        songIdForTrack(track) ||
        `${normalizeKeyPart(track && track.title)}-${normalizeKeyPart(track && track.artist)}`;
      if (!key || seen.has(key)) return;
      seen.add(key);
      added.push(track);
    });

    if (!added.length) {
      if (!opts.silent) {
        setHomeQueueCaption(
          replaceExisting
            ? `No tracks loaded for “${query}”.`
            : `No new tracks added for “${query}”. Queue size: ${existing.length}.`,
        );
      }
      return { addedCount: 0, playIndex: -1 };
    }

    const combinedHomeQueue = [...existing, ...added];
    const overflow = Math.max(0, combinedHomeQueue.length - HOME_QUEUE_MAX_TRACKS);
    const startIndexInCombined = existing.length;
    state.lists.home = combinedHomeQueue.slice(-HOME_QUEUE_MAX_TRACKS);
    if (replaceExisting) {
      state.homePaging.currentPage = 0;
    }
    const playIndex = Math.max(0, startIndexInCombined - overflow);

    renderHomeQueue();
    const actionWord = replaceExisting ? "Loaded" : "Added";
    setHomeQueueCaption(
      `${actionWord} ${added.length} track${added.length === 1 ? "" : "s"} for “${query}”. Queue size: ${state.lists.home.length}.`,
    );

    return { addedCount: added.length, playIndex };
  }

  function setHomeQueueSource(query, endpoint, nextOffset, exhausted) {
    state.homeQueueSource = {
      query: String(query || "").trim(),
      endpoint: String(endpoint || ""),
      nextOffset: Math.max(0, Number(nextOffset) || 0),
      exhausted: Boolean(exhausted),
      loading: false,
      pendingTracks: [],
    };
  }

  async function appendMoreHomeQueueFromSource(autoPlayOnAppend) {
    const source = state.homeQueueSource;
    if (!source || !source.query || !source.endpoint || source.exhausted || source.loading) {
      return false;
    }
    source.loading = true;
    try {
      let pagesTried = 0;
      while (!source.exhausted && pagesTried < HOME_QUEUE_AUTOFETCH_MAX_PAGES) {
        const offset = Math.max(0, Number(source.nextOffset) || 0);
        const payload = await fetchSearchPagePayload(
          source.query,
          source.endpoint,
          offset,
          HOME_QUEUE_BATCH_SIZE,
          null,
          { timeoutMs: HOME_QUEUE_SEARCH_TIMEOUT_MS, retries: 1 },
        );
        source.nextOffset = offset + HOME_QUEUE_BATCH_SIZE;
        pagesTried += 1;

        let tracks = mapSearchPayload(payload);
        if (!tracks.length) {
          source.exhausted = true;
          break;
        }
        if (tracks.length < HOME_QUEUE_BATCH_SIZE) {
          source.exhausted = true;
        }

        await Promise.allSettled(
          tracks.slice(0, Math.min(HOME_QUEUE_BATCH_SIZE, 6)).map(function (track) {
            if (trackHasNonYoutubeCover(track)) return Promise.resolve(track);
            return preResolveTrackCoverIfNeededBounded(track, FOR_YOU_PRERESOLVE_TIMEOUT_MS);
          }),
        );

        const result = appendTracksToHomeQueue(tracks, source.query, { silent: true });
        if (result.addedCount > 0) {
          if (autoPlayOnAppend) {
            void playTrackAt(result.playIndex, "home");
          }
          return true;
        }
      }
      return false;
    } catch (error) {
      return false;
    } finally {
      source.loading = false;
    }
  }

  async function topUpHomeQueueByOneAfterAdvance() {
    const source = state.homeQueueSource;
    if (!source || !source.query || !source.endpoint || source.loading) return false;

    const tryAppendOne = async function () {
      while (Array.isArray(source.pendingTracks) && source.pendingTracks.length) {
        const nextTrack = source.pendingTracks.shift();
        if (!nextTrack) continue;
        if (!trackHasNonYoutubeCover(nextTrack)) {
          await Promise.allSettled([
            preResolveTrackCoverIfNeededBounded(nextTrack, FOR_YOU_PRERESOLVE_TIMEOUT_MS),
          ]);
        }
        const result = appendTracksToHomeQueue([nextTrack], source.query, { silent: true });
        if (result.addedCount > 0) {
          return true;
        }
      }
      return false;
    };

    if (await tryAppendOne()) return true;
    if (source.exhausted) return false;

    source.loading = true;
    try {
      let pagesTried = 0;
      while (!source.exhausted && pagesTried < HOME_QUEUE_AUTOFETCH_MAX_PAGES) {
        const offset = Math.max(0, Number(source.nextOffset) || 0);
        const payload = await fetchSearchPagePayload(
          source.query,
          source.endpoint,
          offset,
          HOME_QUEUE_BATCH_SIZE,
          null,
          { timeoutMs: HOME_QUEUE_SEARCH_TIMEOUT_MS, retries: 1 },
        );
        source.nextOffset = offset + HOME_QUEUE_BATCH_SIZE;
        pagesTried += 1;

        const tracks = mapSearchPayload(payload);
        if (!tracks.length) {
          source.exhausted = true;
          break;
        }
        if (tracks.length < HOME_QUEUE_BATCH_SIZE) {
          source.exhausted = true;
        }
        source.pendingTracks = [...(source.pendingTracks || []), ...tracks];
        if (await tryAppendOne()) return true;
      }
      return false;
    } catch (error) {
      return false;
    } finally {
      source.loading = false;
    }
  }

  function getForYouTotalPages() {
    const size = Math.max(1, Number(state.forYouPaging.pageSize) || FOR_YOU_PAGE_SIZE);
    const total = Array.isArray(state.forYouPaging.tracks)
      ? state.forYouPaging.tracks.length
      : 0;
    if (!total) return 0;
    if (total <= size) return 1;
    // Keep pages visually complete (full rows) by using only full-size pages.
    return Math.max(1, Math.floor(total / size));
  }

  function getForYouPageTracks(pageIndex) {
    const size = Math.max(1, Number(state.forYouPaging.pageSize) || FOR_YOU_PAGE_SIZE);
    const tracks = Array.isArray(state.forYouPaging.tracks)
      ? state.forYouPaging.tracks
      : [];
    const totalPages = getForYouTotalPages();
    if (!totalPages) return [];
    const page = Math.max(0, Math.min(Number(pageIndex) || 0, totalPages - 1));
    const start = page * size;
    return tracks.slice(start, start + size);
  }

  function detectForYouColumns() {
    if (!$.forYouGrid) return 1;
    try {
      const style = window.getComputedStyle($.forYouGrid);
      const template = String(style.gridTemplateColumns || "").trim();
      if (template && template !== "none") {
        const pieces = template.split(/\s+/).filter(function (token) {
          return /px$/.test(token);
        });
        if (pieces.length > 0) return Math.max(1, pieces.length);
      }
    } catch (error) {}
    const width = Number(
      $.forYouGrid.clientWidth ||
        ($.forYouGrid.parentElement && $.forYouGrid.parentElement.clientWidth) ||
        0,
    );
    if (!width) return 1;
    const gap = 16;
    const minCardWidth = 132;
    return Math.max(1, Math.floor((width + gap) / (minCardWidth + gap)));
  }

  function isForYouGridMeasurable() {
    if (!$.forYouGrid) return false;
    if (!$.forYouGrid.isConnected) return false;
    const rects = $.forYouGrid.getClientRects();
    if (!rects || !rects.length) return false;
    return Number($.forYouGrid.clientWidth || 0) > 0;
  }

  function getForYouPageSizeFromLayout() {
    const cols = Math.max(1, detectForYouColumns());
    return Math.max(1, cols * FOR_YOU_ROWS_PER_PAGE);
  }

  function updateForYouPageSize(keepAnchor) {
    if (!isForYouGridMeasurable()) return false;
    const nextSize = getForYouPageSizeFromLayout();
    const prevSize = Math.max(1, Number(state.forYouPaging.pageSize) || FOR_YOU_PAGE_SIZE);
    if (nextSize === prevSize) return false;

    let anchorId = "";
    if (keepAnchor) {
      const currentTracks = getForYouPageTracks(state.forYouPaging.currentPage);
      const anchorTrack = currentTracks.length ? currentTracks[0] : null;
      anchorId = forYouTrackStableId(anchorTrack);
    }

    state.forYouPaging.pageSize = nextSize;

    if (anchorId && Array.isArray(state.forYouPaging.tracks) && state.forYouPaging.tracks.length) {
      const anchorIndex = state.forYouPaging.tracks.findIndex(function (track) {
        return forYouTrackStableId(track) === anchorId;
      });
      if (anchorIndex >= 0) {
        state.forYouPaging.currentPage = Math.floor(anchorIndex / nextSize);
      }
    }

    const totalPages = getForYouTotalPages();
    state.forYouPaging.currentPage = Math.max(
      0,
      Math.min(
        Number(state.forYouPaging.currentPage) || 0,
        Math.max(0, totalPages - 1),
      ),
    );
    return true;
  }

  function setForYouTrackPool(tracks, options) {
    const opts = options || {};
    updateForYouPageSize(true);
    const list = Array.isArray(tracks) ? tracks : [];
    const ordered = pickForYouVisibleTracks(list, list.length || FOR_YOU_PAGE_SIZE);
    const deduped = [];
    const seen = new Set();
    ordered.forEach(function (track) {
      const id =
        forYouTrackStableId(track) ||
        String(songIdForTrack(track) || "").trim() ||
        `${normalizeKeyPart(track && track.title)}-${normalizeKeyPart(track && track.artist)}`;
      if (!id || seen.has(id)) return;
      seen.add(id);
      deduped.push(track);
    });
    state.forYouPaging.tracks = deduped;
    const totalPages = getForYouTotalPages();
    const nextPage = opts.keepPage
      ? Math.max(0, Math.min(Number(state.forYouPaging.currentPage) || 0, Math.max(0, totalPages - 1)))
      : 0;
    state.forYouPaging.currentPage = nextPage;
  }

  async function preResolveForYouPageCovers(pageIndex) {
    const tracks = getForYouPageTracks(pageIndex);
    if (!tracks.length) return;
    const weak = tracks.filter(function (track) {
      return !trackHasNonYoutubeCover(track);
    });
    if (!weak.length) return;
    await Promise.allSettled(
      weak.slice(0, FOR_YOU_PRERESOLVE_MAX_WEAK).map(function (track) {
        return preResolveTrackCoverIfNeededBounded(
          track,
          FOR_YOU_PRERESOLVE_TIMEOUT_MS,
        );
      }),
    );
  }

  function renderForYou() {
    if (!$.forYouGrid) return;
    updateForYouPageSize(true);
    $.forYouGrid.innerHTML = "";
    const pageTracks = getForYouPageTracks(state.forYouPaging.currentPage);
    state.lists.foryou = pageTracks.slice();
    if (!pageTracks.length) {
      renderEmpty($.forYouGrid, "No recommendations available yet.");
      updateForYouStripControls();
      return;
    }
    const frag = document.createDocumentFragment();
    pageTracks.forEach(function (track, index) {
      frag.appendChild(createAvatarCard(track, index, "foryou"));
    });
    $.forYouGrid.appendChild(frag);
    updateSelectedCard();
    syncLikeUI();
    updateForYouStripControls();
    queueLyricsPrefetchForVisibleTracks(
      pageTracks,
      LYRICS_PREFETCH_TRACK_LIMIT_FOR_YOU,
    );
  }

  async function runHomeQueueSearch() {
    const query = String($.homeQuery && $.homeQuery.value ? $.homeQuery.value : "").trim();
    if (!query) return;

    switchView("home");
    setHomeStatus("Building queue...");

    const endpoint = chooseEndpoint(query);
    try {
      let fallbackQueueIsStatic = false;
      const fetchHomePayload = async function (routePath) {
        return fetchSearchPagePayload(
          query,
          routePath,
          0,
          HOME_QUEUE_BATCH_SIZE,
          null,
          { timeoutMs: HOME_QUEUE_SEARCH_TIMEOUT_MS, retries: 1 },
        );
      };
      let activeEndpoint = endpoint;
      let payload = await fetchHomePayload(activeEndpoint);
      let tracks = mapSearchPayload(payload);
      if (!tracks.length && activeEndpoint !== "/search") {
        activeEndpoint = "/search";
        payload = await fetchHomePayload(activeEndpoint);
        tracks = mapSearchPayload(payload);
      }
      if (!tracks.length) {
        const fallback = await fetchSmartSearchFallbackTracks(
          query,
          activeEndpoint,
          null,
        );
        if (fallback.mode === "endpoint" && fallback.endpoint) {
          activeEndpoint = fallback.endpoint;
          tracks = dedupeTracksForSearch(fallback.tracks).slice(
            0,
            HOME_QUEUE_BATCH_SIZE,
          );
          setHomeStatus(
            fallback.message || "No exact match. Building a smart fallback queue.",
          );
        } else if (fallback.mode === "local" && fallback.tracks.length) {
          tracks = fallback.tracks.slice(0, HOME_QUEUE_BATCH_SIZE);
          fallbackQueueIsStatic = true;
          setHomeStatus(
            fallback.message || "No exact match. Building a smart fallback queue.",
          );
        } else {
          setHomeStatus("Could not build queue for that prompt right now.", true);
          return;
        }
      }

      // Upgrade covers for the visible Home queue rows before render when possible.
      await Promise.allSettled(
        tracks.slice(0, Math.min(HOME_QUEUE_BATCH_SIZE, 8)).map(function (track) {
          if (trackHasNonYoutubeCover(track)) return Promise.resolve(track);
          return preResolveTrackCoverIfNeededBounded(track, FOR_YOU_PRERESOLVE_TIMEOUT_MS);
        }),
      );

      const appendResult = appendTracksToHomeQueue(tracks, query, {
        silent: true,
        replaceExisting: true,
      });
      if (!appendResult.addedCount) {
        setHomeStatus("Could not load related tracks for that prompt.");
        setHomeQueueCaption(`Could not load related tracks for “${query}”.`);
        return;
      }
      setHomeQueueSource(
        query,
        activeEndpoint,
        HOME_QUEUE_BATCH_SIZE,
        fallbackQueueIsStatic || tracks.length < HOME_QUEUE_BATCH_SIZE,
      );
      setHomeStatus("");

      // Keep the top header search box in sync for quick refinement in the Search view.
      if ($.query) $.query.value = query;

      void playTrackAt(appendResult.playIndex, "home");
    } catch (error) {
      if (endpoint !== "/search") {
        try {
          const payload = await fetchSearchPagePayload(
            query,
            "/search",
            0,
            HOME_QUEUE_BATCH_SIZE,
            null,
            { timeoutMs: HOME_QUEUE_SEARCH_TIMEOUT_MS, retries: 1 },
          );
          const tracks = mapSearchPayload(payload);
          if (tracks.length) {
            const appendResult = appendTracksToHomeQueue(tracks, query, {
              silent: true,
              replaceExisting: true,
            });
            if (appendResult.addedCount) {
              setHomeQueueSource(query, "/search", HOME_QUEUE_BATCH_SIZE, tracks.length < HOME_QUEUE_BATCH_SIZE);
              setHomeStatus("");
              if ($.query) $.query.value = query;
              void playTrackAt(appendResult.playIndex, "home");
              return;
            }
          }
        } catch (retryError) {
          // Fall through to final message.
        }
      }
      setHomeStatus("Could not build queue from that query. Try Search for a broader result set.", true);
    }
  }

  function updateForYouStripControls() {
    updateForYouPageSize(true);
    const totalPages = getForYouTotalPages();
    const currentPage = Math.max(0, Number(state.forYouPaging.currentPage) || 0);
    const hasPages = totalPages > 0;
    const isLastPage = hasPages && currentPage >= totalPages - 1;
    const pageSize = Math.max(1, Number(state.forYouPaging.pageSize) || FOR_YOU_PAGE_SIZE);
    if ($.forYouPager) {
      $.forYouPager.classList.toggle("hidden", !hasPages);
    }
    if ($.forYouPageLabel) {
      $.forYouPageLabel.textContent = hasPages
        ? `Page ${currentPage + 1} of ${totalPages}`
        : "";
    }
    if ($.forYouPrevBtn) {
      $.forYouPrevBtn.disabled = !hasPages || currentPage <= 0;
    }
    if ($.forYouNextBtn) {
      $.forYouNextBtn.textContent = `Next ${pageSize}`;
      $.forYouNextBtn.disabled =
        !hasPages ||
        state.forYouPaging.loading ||
        (isLastPage && state.forYouPaging.exhausted);
    }
  }

  async function goToForYouPage(pageIndex) {
    let totalPages = getForYouTotalPages();
    if (!totalPages) return;
    const requestedPage = Number(pageIndex) || 0;
    const wantsForward = requestedPage > Number(state.forYouPaging.currentPage || 0);
    if (wantsForward && requestedPage >= totalPages && state.forYouPaging.exhausted) {
      setForYouStatus("No more unique recommendations right now. Try Refresh All for a new mix.");
    }
    if (wantsForward && requestedPage >= totalPages && !state.forYouPaging.exhausted) {
      await loadForYou(true, {
        avoidCurrentBatch: true,
        appendPool: true,
      });
      totalPages = getForYouTotalPages();
    }
    if (!totalPages) return;
    const nextPage = Math.max(0, Math.min(requestedPage, totalPages - 1));
    if (nextPage === state.forYouPaging.currentPage) {
      updateForYouStripControls();
      return;
    }
    state.forYouPaging.currentPage = nextPage;
    renderForYou();
    await preResolveForYouPageCovers(nextPage);
    if (nextPage === state.forYouPaging.currentPage) {
      renderForYou();
    }
  }

  async function replayCurrentTrackAfterEnd(providerHint) {
    if (!state.current) return false;
    const provider = String(providerHint || state.activePlaybackProvider || "")
      .trim()
      .toLowerCase();
    try {
      if (
        provider === "spotify" &&
        state.spotify.player &&
        typeof state.spotify.player.seek === "function"
      ) {
        await state.spotify.player.seek(0);
        if (typeof state.spotify.player.resume === "function") {
          await state.spotify.player.resume();
        } else if (typeof state.spotify.player.togglePlay === "function") {
          await state.spotify.player.togglePlay();
        }
      } else if (state.yt && typeof state.yt.loadVideoById === "function") {
        const videoId = String(state.current.videoId || "").trim();
        if (!isYoutubeId(videoId)) return false;
        state.yt.loadVideoById(videoId);
        state.yt.playVideo();
      } else {
        return false;
      }
      setPlayState(true);
      startProgress();
      updateProgress(true);
      return true;
    } catch (error) {
      return false;
    }
  }

  function trackHasStrongCover(track) {
    const candidates = Array.isArray(track && track.coverCandidates)
      ? track.coverCandidates
      : [];
    if (!candidates.length) return false;
    return candidates.some(function (url) {
      const value = String(url || "");
      return value && !isLocalFallbackCoverUrl(value);
    });
  }

  function trackHasNonYoutubeCover(track) {
    const candidates = Array.isArray(track && track.coverCandidates)
      ? track.coverCandidates
      : [];
    if (!candidates.length) return false;
    return candidates.some(function (url) {
      const value = String(url || "");
      if (!value || isLocalFallbackCoverUrl(value)) return false;
      return !isYoutubeThumbUrl(value);
    });
  }

  function pickForYouVisibleTracks(tracks, limit) {
    const max = Math.max(1, Number(limit) || FOR_YOU_VISIBLE_LIMIT);
    const safe = Array.isArray(tracks) ? tracks : [];
    const strong = [];
    const weak = [];
    safe.forEach(function (track) {
      if (trackHasStrongCover(track)) strong.push(track);
      else weak.push(track);
    });
    return [...strong, ...weak].slice(0, max);
  }

  function getForYouVisibleLimit() {
    const rowCap = Math.max(1, Number(FOR_YOU_TARGET_ROWS) || 3);
    const hardCap = Math.max(1, Number(FOR_YOU_VISIBLE_LIMIT) || 24);
    if (!$.forYouGrid) return hardCap;
    const width = Number(
      $.forYouGrid.clientWidth ||
        ($.forYouGrid.parentElement && $.forYouGrid.parentElement.clientWidth) ||
        0,
    );
    if (!width) return hardCap;
    const minCardWidth = 132;
    const gap = 16;
    const columns = Math.max(1, Math.floor((width + gap) / (minCardWidth + gap)));
    let rows = rowCap;
    try {
      const gridRect = $.forYouGrid.getBoundingClientRect();
      const viewportH = Math.max(
        Number(window.innerHeight) || 0,
        Number(document.documentElement && document.documentElement.clientHeight) || 0,
      );
      const bottomReserve = 72; // page breathing room
      const availableH = Math.max(0, viewportH - gridRect.top - bottomReserve);
      const estimatedRowH = 176; // cover + captions + row gap
      if (availableH > 0) {
        rows = Math.max(1, Math.min(rowCap, Math.floor(availableH / estimatedRowH)));
      }
    } catch (error) {
      // Keep rowCap if layout metrics are unavailable.
    }
    return Math.max(1, Math.min(hardCap, columns * rows));
  }

  function forYouTrackStableId(track) {
    const titleKey = normalizeKeyPart(track && track.title);
    const artistKey = normalizeKeyPart(track && track.artist);
    if (titleKey || artistKey) {
      return `ta:${titleKey}|${artistKey}`;
    }
    const primary = String(songIdForTrack(track) || "").trim();
    if (primary) return `id:${primary}`;
    return "";
  }

  function shuffleTracksCopy(tracks) {
    const arr = Array.isArray(tracks) ? tracks.slice() : [];
    for (let i = arr.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  function readForYouSeenIds() {
    try {
      const raw = localStorage.getItem(FOR_YOU_SEEN_IDS_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .map(function (v) {
          return String(v || "").trim();
        })
        .filter(Boolean)
        .slice(-FOR_YOU_SEEN_IDS_MAX);
    } catch (error) {
      return [];
    }
  }

  function writeForYouSeenIds(ids) {
    try {
      const safe = (Array.isArray(ids) ? ids : [])
        .map(function (v) {
          return String(v || "").trim();
        })
        .filter(Boolean)
        .slice(-FOR_YOU_SEEN_IDS_MAX);
      localStorage.setItem(FOR_YOU_SEEN_IDS_KEY, JSON.stringify(safe));
    } catch (error) {
      return;
    }
  }

  function pickRotatingForYouTracks(tracks, limit, options) {
    const max = Math.max(1, Number(limit) || FOR_YOU_VISIBLE_LIMIT);
    const opts = options || {};
    const avoidIds = new Set(
      (Array.isArray(opts.avoidIds) ? opts.avoidIds : [])
        .map(function (v) {
          return String(v || "").trim();
        })
        .filter(Boolean),
    );
    const orderedRaw = pickForYouVisibleTracks(
      Array.isArray(tracks) ? tracks : [],
      Array.isArray(tracks) ? tracks.length : max,
    );
    const ordered = [];
    const poolSeen = new Set();
    orderedRaw.forEach(function (track) {
      const id = forYouTrackStableId(track);
      if (id && poolSeen.has(id)) return;
      if (id) poolSeen.add(id);
      ordered.push(track);
    });
    if (!ordered.length) return [];

    const unblockedOrdered = ordered.filter(function (track) {
      const id = forYouTrackStableId(track);
      return !id || !avoidIds.has(id);
    });
    const candidateOrdered = unblockedOrdered.length ? unblockedOrdered : ordered;

    let seenList = readForYouSeenIds();
    let seenSet = new Set(seenList);
    let selected = candidateOrdered.filter(function (track) {
      const id = forYouTrackStableId(track);
      return !id || !seenSet.has(id);
    });

    if (selected.length < max) {
      // Prefer unseen tracks first, then top up with previously seen tracks without
      // immediately resetting the entire rotation cycle.
      if (!selected.length) {
        // Fully exhausted the pool; reset and start a fresh cycle.
        seenList = [];
        seenSet = new Set();
        selected = candidateOrdered.slice();
      } else {
        const selectedSet = new Set(
          selected
            .map(function (track) {
              return forYouTrackStableId(track);
            })
            .filter(Boolean),
        );
        const topped = selected.slice();
        for (let i = 0; i < candidateOrdered.length && topped.length < max; i += 1) {
          const track = candidateOrdered[i];
          const id = forYouTrackStableId(track);
          if (id && selectedSet.has(id)) continue;
          topped.push(track);
          if (id) selectedSet.add(id);
        }
        selected = topped;
      }
    }

    const picked = selected.slice(0, max);
    picked.forEach(function (track) {
      const id = forYouTrackStableId(track);
      if (!id || seenSet.has(id)) return;
      seenSet.add(id);
      seenList.push(id);
    });
    writeForYouSeenIds(seenList);
    return picked;
  }

  function markForYouSeenTracks(tracks) {
    const list = Array.isArray(tracks) ? tracks : [];
    if (!list.length) return;
    let seenList = readForYouSeenIds();
    const seenSet = new Set(seenList);
    let changed = false;
    list.forEach(function (track) {
      const id = forYouTrackStableId(track);
      if (!id || seenSet.has(id)) return;
      seenSet.add(id);
      seenList.push(id);
      changed = true;
    });
    if (changed) writeForYouSeenIds(seenList);
  }

  async function preResolveTrackCoverIfNeeded(track) {
    if (!track || trackHasNonYoutubeCover(track)) return track;
    const resolved = await resolveCoverCandidatesWithYtDlp(track);
    if (!resolved) return track;
    const mergedCandidates = prioritizeCoverCandidates([
      ...(Array.isArray(resolved.coverCandidates) ? resolved.coverCandidates : []),
      ...(Array.isArray(track.coverCandidates) ? track.coverCandidates : []),
    ]);
    track.coverCandidates = mergedCandidates;
    return track;
  }

  async function preResolveTrackCoverIfNeededBounded(track, timeoutMs) {
    const budget = Math.max(250, Number(timeoutMs) || 0);
    try {
      return await Promise.race([
        preResolveTrackCoverIfNeeded(track),
        sleep(budget).then(function () {
          return track;
        }),
      ]);
    } catch (error) {
      return track;
    }
  }

  async function preResolveVisibleForYouTracks(selectedTracks, allTracks, limit) {
    const max = Math.max(1, Number(limit) || FOR_YOU_VISIBLE_LIMIT);
    const selected = Array.isArray(selectedTracks) ? selectedTracks.slice() : [];
    if (!selected.length) return [];

    const weakSelected = selected.filter(function (track) {
      return !trackHasStrongCover(track);
    });
    if (weakSelected.length) {
      await Promise.allSettled(
        weakSelected.slice(0, FOR_YOU_PRERESOLVE_MAX_WEAK).map(function (track) {
          return preResolveTrackCoverIfNeededBounded(
            track,
            FOR_YOU_PRERESOLVE_TIMEOUT_MS,
          );
        }),
      );
    }

    const out = [];
    const used = new Set();
    selected.forEach(function (track) {
      const id = forYouTrackStableId(track);
      if (id) used.add(id);
      if (trackHasStrongCover(track) && out.length < max) {
        out.push(track);
      }
    });

    if (out.length >= max) return out.slice(0, max);

    const pool = pickForYouVisibleTracks(
      Array.isArray(allTracks) ? allTracks : [],
      Array.isArray(allTracks) ? allTracks.length : max,
    );
    const extras = [];
    for (let i = 0; i < pool.length && out.length + extras.length < max; i += 1) {
      const track = pool[i];
      const id = forYouTrackStableId(track);
      if (id && used.has(id)) continue;
      if (trackHasStrongCover(track)) {
        extras.push(track);
        if (id) used.add(id);
      }
    }

    const weakPool = [];
    if (out.length + extras.length < max) {
      for (let i = 0; i < pool.length && out.length + extras.length + weakPool.length < max; i += 1) {
        const track = pool[i];
        const id = forYouTrackStableId(track);
        if (id && used.has(id)) continue;
        if (!trackHasStrongCover(track)) {
          weakPool.push(track);
          if (id) used.add(id);
        }
      }
      if (weakPool.length) {
        await Promise.allSettled(
          weakPool.slice(0, FOR_YOU_PRERESOLVE_MAX_WEAK).map(function (track) {
            return preResolveTrackCoverIfNeededBounded(
              track,
              FOR_YOU_PRERESOLVE_TIMEOUT_MS,
            );
          }),
        );
        weakPool.forEach(function (track) {
          if (trackHasStrongCover(track) && out.length + extras.length < max) {
            extras.push(track);
          }
        });
      }
    }

    const finalTracks = [...out, ...extras].slice(0, max);
    if (!finalTracks.length && selected.length) {
      // Do not blank the whole row if cover pre-resolve timed out; render and let per-image resolver recover.
      return selected.slice(0, max);
    }
    markForYouSeenTracks(finalTracks);
    return finalTracks;
  }

  function renderPlayerTrackMeta(track) {
    if ($.player) {
      $.player.classList.toggle("has-track", !!track);
    }
    syncHomePlaybackChromeVisibility();
    if ($.playerDescription) {
      $.playerDescription.textContent = shortDescription(
        track && track.description,
      );
    }

    // Update Right Panel Now Playing info
    if ($.rightPlayerArt && track) {
      setImgWithFallback(
        $.rightPlayerArt,
        prioritizeCoverCandidates([
          ...(Array.isArray(track.coverCandidates) ? track.coverCandidates : []),
          youtubeThumb(track.videoId, "maxresdefault"),
          youtubeThumb(track.videoId, "hq720"),
          youtubeThumb(track.videoId, "sddefault"),
          youtubeThumb(track.videoId, "hqdefault"),
          youtubeThumb(track.videoId, "mqdefault"),
          youtubeThumb(track.videoId, "default"),
        ]),
        PLAYER_PLACEHOLDER,
        {
          track,
          allowYtDlpResolve: false,
        },
      );
    }
    if ($.rightPlayerTitle && track) {
      $.rightPlayerTitle.textContent = track.title || "Unknown Title";
    }
    if ($.rightPlayerArtist && track) {
      $.rightPlayerArtist.textContent = track.artist || "Unknown Artist";
    }

    syncLikeUI();
  }

  function mergeTrackMeta(track, payload) {
    if (!track || !payload) return track;
    const merged = { ...track };
    if (payload.description) {
      merged.description = shortDescription(payload.description);
    }
    const instrumental = toOptionalBool(payload.instrumental);
    if (instrumental !== null) {
      merged.instrumental = instrumental;
    }
    if (payload.instrumental_confidence !== undefined) {
      merged.instrumentalConfidence = Number(
        payload.instrumental_confidence || 0,
      );
    }
    return merged;
  }

  async function hydrateCurrentTrackMeta() {
    if (!state.current) return;
    const key = trackCacheKey(state.current);
    const cached = state.enrichCache.get(key);
    if (cached) {
      state.current = mergeTrackMeta(state.current, cached);
      renderPlayerTrackMeta(state.current);
      const list = listFor(state.activeListKey);
      if (state.queueIndex >= 0 && state.queueIndex < list.length) {
        list[state.queueIndex] = { ...state.current };
      }
      return;
    }
    if (
      state.current.description &&
      toOptionalBool(state.current.instrumental) !== null
    )
      return;

    const params = new URLSearchParams({
      track_id: state.current.trackId || state.current.videoId || "",
      title: state.current.title || "",
      artist: state.current.artist || "",
    });
    try {
      const payload = await requestJSON(
        `/api/track_enrich?${params.toString()}`,
        {},
        { timeoutMs: 8500, retries: 1 },
      );
      state.enrichCache.set(key, payload);
      if (!state.current || key !== trackCacheKey(state.current)) return;
      state.current = mergeTrackMeta(state.current, payload);
      renderPlayerTrackMeta(state.current);
      const list = listFor(state.activeListKey);
      if (state.queueIndex >= 0 && state.queueIndex < list.length) {
        list[state.queueIndex] = { ...state.current };
      }
    } catch (error) {
      return;
    }
  }

  function parseLRC(text) {
    const lines = String(text || "")
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n")
      .split("\n");
    const parsed = [];
    const timeTag = /\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]/g;

    for (let i = 0; i < lines.length; i += 1) {
      const rawLine = String(lines[i] || "").trim();
      if (!rawLine) continue;

      timeTag.lastIndex = 0;
      const times = [];
      let match;
      while ((match = timeTag.exec(rawLine)) !== null) {
        const min = parseInt(match[1], 10) || 0;
        const sec = parseInt(match[2], 10) || 0;
        const fracText = match[3] ? String(match[3]) : "";
        const frac = fracText
          ? parseInt(fracText.padEnd(3, "0").slice(0, 3), 10) / 1000
          : 0;
        times.push(min * 60 + sec + frac);
      }
      if (!times.length) continue;

      const content = rawLine.replace(timeTag, "").trim();
      if (!content) continue;

      for (let t = 0; t < times.length; t += 1) {
        parsed.push({ time: times[t], content });
      }
    }

    return parsed.sort((a, b) => a.time - b.time);
  }

  async function hydrateLyrics(track) {
    if (!track) return;
    const requestId = ++state.lyricsRequestId;
    const loaderStartedAt =
      window.performance && typeof window.performance.now === "function"
        ? window.performance.now()
        : Date.now();
    const isStaleLyricsRequest = function () {
      return (
        requestId !== state.lyricsRequestId ||
        !state.current ||
        trackCacheKey(track) !== trackCacheKey(state.current)
      );
    };
    const ensureLoaderVisible = async function () {
      const now =
        window.performance && typeof window.performance.now === "function"
          ? window.performance.now()
          : Date.now();
      const elapsed = Math.max(0, now - loaderStartedAt);
      const minLoaderMs = 420;
      if (elapsed < minLoaderMs) {
        await sleep(minLoaderMs - elapsed);
      }
    };
    state.lyricsData = null;
    state.lyricsLineEls = null;
    state.lyricsMode = "none";
    state.activeLyricIndex = -1;
    if ($.lyricsContent) {
      $.lyricsContent.innerHTML =
        "<div class='lyrics-loading' role='status' aria-live='polite'><span class='lyrics-loading-text'>Loading lyrics</span><span class='lyrics-loading-dots'><span class='lyrics-loading-dot'></span><span class='lyrics-loading-dot'></span><span class='lyrics-loading-dot'></span></span></div>";
    }
    if ($.lyricsScroll) $.lyricsScroll.scrollTop = 0;

    const params = new URLSearchParams({
      title: track.title || "",
      artist: track.artist || "",
    });

    try {
      const payload = await requestJSON(
        `/api/lyrics?${params.toString()}`,
        {},
        { timeoutMs: 30000, retries: 0 },
      );
      if (isStaleLyricsRequest()) return;
      await ensureLoaderVisible();
      if (isStaleLyricsRequest()) return;
      const lyricsText = String(payload && payload.lyrics ? payload.lyrics : "")
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n")
        .trim();
      const wantsSynced = Boolean(payload && payload.synced);

      if (!lyricsText) {
        if ($.lyricsContent) {
          $.lyricsContent.innerHTML =
            "<div class='lyric-line'>No lyrics found for this track.</div>";
        }
        return;
      }

      if (wantsSynced) {
        const parsed = parseLRC(lyricsText);
        if (parsed.length > 0) {
          state.lyricsMode = "synced";
          state.lyricsData = parsed;
          if ($.lyricsContent) {
            $.lyricsContent.innerHTML = "";
            const frag = document.createDocumentFragment();
            parsed.forEach((line, index) => {
              const el = document.createElement("div");
              el.className = "lyric-line";
              el.textContent = line.content;
              el.dataset.index = String(index);
              el.dataset.time = String(line.time);
              el.addEventListener("click", () => {
                const timing = getActivePlaybackTiming();
                const duration = Number(timing && timing.duration ? timing.duration : 0);
                if (!duration) return;
                seekTo((Number(line.time) / duration) * 100);
              });
              frag.appendChild(el);
            });
            $.lyricsContent.appendChild(frag);
            state.lyricsLineEls = Array.from($.lyricsContent.children);
          }
          return;
        }
      }

      // Plain (unsynced) lyrics fallback.
      state.lyricsMode = "plain";
      if ($.lyricsContent) {
        $.lyricsContent.innerHTML = "";
        const frag = document.createDocumentFragment();
        let added = 0;
        lyricsText.split("\n").forEach((line) => {
          const content = String(line || "").trim();
          if (!content) return;
          const el = document.createElement("div");
          el.className = "lyric-line";
          el.textContent = content;
          frag.appendChild(el);
          added += 1;
        });
        if (!added) {
          $.lyricsContent.innerHTML =
            "<div class='lyric-line'>No lyrics found for this track.</div>";
          return;
        }
        $.lyricsContent.appendChild(frag);
      }
    } catch (e) {
      if (isStaleLyricsRequest()) return;
      await ensureLoaderVisible();
      if (isStaleLyricsRequest()) return;
      if ($.lyricsContent) {
        $.lyricsContent.innerHTML =
          "<div class='lyric-line'>Could not load lyrics.</div>";
      }
    }
  }

  function findLyricIndexAtTime(current) {
    const data = state.lyricsData;
    if (!data || !data.length) return -1;
    const target = Number(current || 0) + 0.3; // small lead so "active" feels natural
    let lo = 0;
    let hi = data.length - 1;
    let ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (data[mid].time <= target) {
        ans = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return ans;
  }

  function scrollLyricsToIndex(index, behavior) {
    if (!$.lyricsScroll) return;
    if (!state.lyricsLineEls || !state.lyricsLineEls.length) return;
    if (!Number.isInteger(index) || index < 0) {
      $.lyricsScroll.scrollTop = 0;
      return;
    }
    const el = state.lyricsLineEls[index];
    if (!el) return;
    const top = Math.max(0, el.offsetTop - $.lyricsScroll.clientHeight * 0.4);
    $.lyricsScroll.scrollTo({ top, behavior: behavior || "smooth" });
  }

  function applyActiveLyric(newIndex, forceFullRefresh) {
    const lines = state.lyricsLineEls;
    if (!lines || !lines.length) return;

    const oldIndex = state.activeLyricIndex;
    if (newIndex === oldIndex && !forceFullRefresh) return;

    const fullRefresh =
      forceFullRefresh ||
      oldIndex === -1 ||
      newIndex < oldIndex ||
      newIndex < 0;

    if (fullRefresh) {
      for (let i = 0; i < lines.length; i += 1) {
        const isActive = i === newIndex;
        lines[i].classList.toggle("active", isActive);
        lines[i].classList.toggle("past", newIndex >= 0 && i < newIndex);
      }
    } else if (newIndex > oldIndex) {
      if (oldIndex >= 0 && lines[oldIndex]) {
        lines[oldIndex].classList.remove("active");
        lines[oldIndex].classList.add("past");
      }
      for (let i = oldIndex + 1; i < newIndex; i += 1) {
        if (lines[i]) lines[i].classList.add("past");
      }
      if (lines[newIndex]) {
        lines[newIndex].classList.add("active");
        lines[newIndex].classList.remove("past");
      }
    }

    state.activeLyricIndex = newIndex;
    scrollLyricsToIndex(newIndex, fullRefresh ? "auto" : "smooth");
  }

  function getSpotifyPlaybackTiming() {
    const snap = state.spotify && state.spotify.progressSnapshot;
    if (!snap) return { current: 0, duration: 0, provider: "spotify" };
    const durationMs = Math.max(0, Number(snap.durationMs || 0));
    let currentMs = Math.max(0, Number(snap.positionMs || 0));
    const updatedAt = Number(snap.updatedAt || 0);
    if (!snap.paused && updatedAt > 0) {
      currentMs += Math.max(0, Date.now() - updatedAt);
    }
    if (durationMs > 0) {
      currentMs = Math.min(currentMs, durationMs);
    }
    return {
      current: currentMs / 1000,
      duration: durationMs / 1000,
      provider: "spotify",
    };
  }

  function getActivePlaybackTiming() {
    if (state.activePlaybackProvider === "spotify") {
      return getSpotifyPlaybackTiming();
    }
    if (state.yt && typeof state.yt.getDuration === "function") {
      const duration = Number(state.yt.getDuration() || 0);
      const current = Number(
        state.yt && typeof state.yt.getCurrentTime === "function"
          ? state.yt.getCurrentTime() || 0
          : 0,
      );
      return { current, duration, provider: "youtube" };
    }
    return { current: 0, duration: 0, provider: String(state.activePlaybackProvider || "youtube") };
  }

  function updateProgress(force) {
    const timing = getActivePlaybackTiming();
    const duration = Number(timing && timing.duration ? timing.duration : 0);
    const current = Number(timing && timing.current ? timing.current : 0);
    if (!duration) return;

    if (!state.isSeeking || force) {
      const pct = Math.max(0, Math.min(100, (current / duration) * 100));
      if ($.seek) {
        $.seek.value = String(pct);
        updateRangeFill($.seek);
      }
      if ($.timeCurrent) $.timeCurrent.textContent = formatTime(current);

      // --- Sync Lyrics ---
      if (
        state.lyricsMode === "synced" &&
        state.lyricsData &&
        state.lyricsData.length > 0
      ) {
        let newIndex = state.activeLyricIndex;

        // Normal playback: advance forward using a cursor. Seeking/backwards: fall back to binary search.
        if (newIndex >= 0 && current + 0.3 < state.lyricsData[newIndex].time) {
          newIndex = findLyricIndexAtTime(current);
          applyActiveLyric(newIndex, true);
        } else {
          if (newIndex < 0) newIndex = findLyricIndexAtTime(current);
          while (
            newIndex + 1 < state.lyricsData.length &&
            current + 0.3 >= state.lyricsData[newIndex + 1].time
          ) {
            newIndex += 1;
          }
          if (newIndex !== state.activeLyricIndex) {
            applyActiveLyric(newIndex, false);
          }
        }
      }
    }
    if ($.timeTotal) $.timeTotal.textContent = formatTime(duration);
  }

  function setPlayState(playing) {
    state.isPlaying = playing;
    if ($.player) $.player.classList.toggle("active", !!playing);
    if (!$.playBtn) return;
    const playIcon = $.playBtn.querySelector(".play-icon");
    const pauseIcon = $.playBtn.querySelector(".pause-icon");
    if (playIcon && pauseIcon) {
      playIcon.style.display = playing ? "none" : "";
      pauseIcon.style.display = playing ? "" : "none";
    }
    $.playBtn.setAttribute("aria-label", playing ? "Pause" : "Play");
  }

  function startProgress() {
    stopProgress();
    state.progressTimer = setInterval(function () {
      updateProgress(false);
    }, 250);
  }

  function stopProgress() {
    if (!state.progressTimer) return;
    clearInterval(state.progressTimer);
    state.progressTimer = null;
  }

  function setVolume(value) {
    if (state.yt && typeof state.yt.setVolume === "function") {
      state.yt.setVolume(Number(value));
    }
    if (state.spotify.player && typeof state.spotify.player.setVolume === "function") {
      try {
        void state.spotify.player.setVolume(
          Math.max(0, Math.min(1, Number(value || 0) / 100)),
        );
      } catch (error) {}
    }
  }

  function seekTo(percent) {
    const timing = getActivePlaybackTiming();
    const duration = Number(timing && timing.duration ? timing.duration : 0);
    if (!duration) return;
    const targetSeconds = duration * (Number(percent) / 100);
    if (
      state.activePlaybackProvider === "spotify" &&
      state.spotify.player &&
      typeof state.spotify.player.seek === "function"
    ) {
      try {
        void state.spotify.player.seek(Math.round(targetSeconds * 1000));
      } catch (error) {}
      return;
    }
    if (state.yt && typeof state.yt.seekTo === "function") {
      state.yt.seekTo(targetSeconds, true);
    }
  }

  function normalizeVideoCandidateIds(values) {
    const out = [];
    const seen = new Set();
    (Array.isArray(values) ? values : []).forEach(function (value) {
      const id = String(value || "").trim();
      if (!isYoutubeId(id) || seen.has(id)) return;
      seen.add(id);
      out.push(id);
    });
    return out;
  }

  function setCurrentVideoCandidates(values, activeId) {
    if (!state.current) return;
    const normalized = normalizeVideoCandidateIds(values);
    const unblocked = normalized.filter(function (id) {
      return !isVideoIdEmbedBlocked(id);
    });
    const preferred = String(activeId || state.current.videoId || "").trim();
    let ordered = unblocked.length ? unblocked : normalized;
    if (isYoutubeId(preferred)) {
      ordered = normalizeVideoCandidateIds([preferred, ...normalized]);
      const orderedUnblocked = ordered.filter(function (id) {
        return !isVideoIdEmbedBlocked(id);
      });
      if (orderedUnblocked.length) {
        ordered = orderedUnblocked;
      }
    }
    state.current.videoCandidateIds = ordered;
    state.current.videoCandidateIndex = ordered.findIndex(function (id) {
      return id === preferred;
    });
    if (state.current.videoCandidateIndex < 0) {
      state.current.videoCandidateIndex = ordered.length ? 0 : -1;
    }
    if (
      ordered.length &&
      (!isYoutubeId(state.current.videoId) ||
        !ordered.includes(String(state.current.videoId || "")))
    ) {
      state.current.videoId = ordered[state.current.videoCandidateIndex];
    }
  }

  function advanceToNextVideoCandidate() {
    if (!state.current) return false;
    const candidates = normalizeVideoCandidateIds(state.current.videoCandidateIds);
    if (!candidates.length) return false;
    let index = Number(state.current.videoCandidateIndex);
    if (!Number.isInteger(index) || index < 0) {
      index = candidates.findIndex(function (id) {
        return id === state.current.videoId;
      });
    }
    let nextIndex = index + 1;
    while (nextIndex >= 0 && nextIndex < candidates.length) {
      if (!isVideoIdEmbedBlocked(candidates[nextIndex])) break;
      nextIndex += 1;
    }
    if (nextIndex < 0 || nextIndex >= candidates.length) return false;
    state.current.videoCandidateIds = candidates;
    state.current.videoCandidateIndex = nextIndex;
    state.current.videoId = candidates[nextIndex];
    const list = listFor(state.activeListKey);
    if (state.queueIndex >= 0 && state.queueIndex < list.length) {
      list[state.queueIndex] = { ...state.current };
    }
    return true;
  }

  async function resolveTrackSource(forceLookup) {
    if (!state.current) return false;
    if (
      !forceLookup &&
      isYoutubeId(state.current.videoId) &&
      !isVideoIdEmbedBlocked(state.current.videoId)
    ) {
      return true;
    }
    // Do not consume the one recovery attempt during the initial pre-play lookup.
    if (forceLookup) {
      if (state.fallbackTried) return false;
      state.fallbackTried = true;
    }

    try {
      postPlayerDebug("resolve_video_start", {
        track: {
          title: state.current.title,
          artist: state.current.artist,
          videoId: state.current.videoId || "",
        },
        note: forceLookup ? "forceLookup" : "initialLookup",
        candidates: state.current.videoCandidateIds || [],
        candidate_index: state.current.videoCandidateIndex,
      });
      const params = new URLSearchParams({
        track_id: state.current.trackId || state.current.videoId || "",
        video_id: state.current.videoId || "",
        title: state.current.title || "",
        artist: state.current.artist || "",
      });
      const data = await requestJSON(
        `/api/resolve_video?${params.toString()}`,
        {},
        { timeoutMs: 15000, retries: 1 },
      );
      if (!data.video_id || !isYoutubeId(data.video_id)) return false;
      state.current.videoId = String(data.video_id);
      setCurrentVideoCandidates(
        Array.isArray(data.video_candidates) ? data.video_candidates : [data.video_id],
        data.video_id,
      );

      const list = listFor(state.activeListKey);
      if (state.queueIndex >= 0 && state.queueIndex < list.length) {
        list[state.queueIndex] = { ...state.current };
      }
      postPlayerDebug("resolve_video_ok", {
        track: {
          title: state.current.title,
          artist: state.current.artist,
          videoId: state.current.videoId || "",
        },
        note: String(data.provider || ""),
        candidates: state.current.videoCandidateIds || [],
        candidate_index: state.current.videoCandidateIndex,
      });
      return true;
    } catch (error) {
      postPlayerDebug("resolve_video_fail", {
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId || "",
            }
          : null,
        note: String((error && error.message) || error || ""),
      });
      return false;
    }
  }

  async function resolveFallback(errorEvent) {
    if (state.fallbackInFlight) return;
    state.fallbackInFlight = true;
    try {
    const ytErrorCode = Number(
      errorEvent && typeof errorEvent.data !== "undefined" ? errorEvent.data : NaN,
    );
    const embedBlockedError = isEmbedBlockedYoutubeErrorCode(ytErrorCode);
    if (embedBlockedError && state.current && isYoutubeId(state.current.videoId)) {
      markVideoIdEmbedBlocked(state.current.videoId, ytErrorCode);
      state.current.embedBlockedAttempts = Number(state.current.embedBlockedAttempts || 0) + 1;
    }
    if (!Number.isNaN(ytErrorCode)) {
      // Helpful for debugging embed-restricted / unavailable videos (e.g. 100, 101, 150).
      console.warn("[YouTube Player] onError", {
        code: ytErrorCode,
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId,
              candidates: state.current.videoCandidateIds,
            }
          : null,
      });
    }
    postPlayerDebug("yt_onerror", {
      code: Number.isNaN(ytErrorCode) ? null : ytErrorCode,
      track: state.current
        ? {
            title: state.current.title,
            artist: state.current.artist,
            videoId: state.current.videoId || "",
          }
        : null,
      candidates: state.current ? state.current.videoCandidateIds || [] : [],
      candidate_index: state.current ? state.current.videoCandidateIndex : null,
    });
    const lockedVideoId =
      state.current && STRICT_METADATA_VIDEO_ID_PLAYBACK
        ? String(state.current.lockedPlaybackVideoId || "")
        : "";
    const isLockedOriginalFailure =
      embedBlockedError &&
      state.current &&
      isYoutubeId(lockedVideoId) &&
      String(state.current.videoId || "") === lockedVideoId;
    if (isLockedOriginalFailure) {
      if (await attemptSpotifyFallbackAfterYoutubeFailure(ytErrorCode)) {
        return;
      }
      const codeSuffix = Number.isNaN(ytErrorCode) ? "" : ` (YouTube error ${ytErrorCode})`;
      setStatus(`Original upload is embed-blocked${codeSuffix}.`, true);
      postPlayerDebug("fallback_locked_video_id_blocked", {
        code: Number.isNaN(ytErrorCode) ? null : ytErrorCode,
        track: {
          title: state.current.title,
          artist: state.current.artist,
          videoId: state.current.videoId || "",
        },
        note: "strict_metadata_video_id",
      });
      return;
    }
    if (
      embedBlockedError &&
      state.current &&
      Number(state.current.embedBlockedAttempts || 0) >= YT_EMBED_BLOCK_MAX_ATTEMPTS_PER_TRACK
    ) {
      if (await attemptSpotifyFallbackAfterYoutubeFailure(ytErrorCode)) {
        return;
      }
      const codeSuffix = ` (YouTube error ${ytErrorCode})`;
      setStatus(`Embed-blocked on multiple uploads${codeSuffix}.`, true);
      postPlayerDebug("fallback_quick_fail_embed_block", {
        code: ytErrorCode,
        track: {
          title: state.current.title,
          artist: state.current.artist,
          videoId: state.current.videoId || "",
        },
        candidates: state.current.videoCandidateIds || [],
        candidate_index: state.current.videoCandidateIndex,
        note: `attempts=${state.current.embedBlockedAttempts}`,
      });
      return;
    }
    if (advanceToNextVideoCandidate()) {
      postPlayerDebug("fallback_advance_local", {
        code: Number.isNaN(ytErrorCode) ? null : ytErrorCode,
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId || "",
            }
          : null,
        candidates: state.current ? state.current.videoCandidateIds || [] : [],
        candidate_index: state.current ? state.current.videoCandidateIndex : null,
      });
      if (
        state.yt &&
        typeof state.yt.loadVideoById === "function" &&
        state.current &&
        isYoutubeId(state.current.videoId)
      ) {
        state.yt.loadVideoById(state.current.videoId);
        state.yt.playVideo();
        setStatus("");
        return;
      }
    }
    const failedVideoId =
      state.current && isYoutubeId(state.current.videoId)
        ? String(state.current.videoId)
        : "";
    const ok = await resolveTrackSource(true);
    if (!ok) {
      if (await attemptSpotifyFallbackAfterYoutubeFailure(ytErrorCode)) {
        return;
      }
      const codeSuffix = Number.isNaN(ytErrorCode) ? "" : ` (YouTube error ${ytErrorCode})`;
      setStatus(`Cannot play this track right now${codeSuffix}.`, true);
      postPlayerDebug("fallback_resolve_fail", {
        code: Number.isNaN(ytErrorCode) ? null : ytErrorCode,
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId || "",
            }
          : null,
      });
      return;
    }
    // The current video just failed. If we fetched new candidate IDs and the failed ID is still first,
    // skip it immediately instead of replaying the same broken/blocked source.
    if (failedVideoId && state.current) {
      const candidates = normalizeVideoCandidateIds(state.current.videoCandidateIds);
      const currentId = String(state.current.videoId || "");
      if (candidates.length > 1 && currentId === failedVideoId) {
        const failedIdx = candidates.findIndex(function (id) {
          return id === failedVideoId;
        });
        state.current.videoCandidateIds = candidates;
        state.current.videoCandidateIndex = failedIdx >= 0 ? failedIdx : 0;
        if (advanceToNextVideoCandidate()) {
          postPlayerDebug("fallback_advance_after_resolve", {
            code: Number.isNaN(ytErrorCode) ? null : ytErrorCode,
            track: state.current
              ? {
                  title: state.current.title,
                  artist: state.current.artist,
                  videoId: state.current.videoId || "",
                }
              : null,
            candidates: state.current ? state.current.videoCandidateIds || [] : [],
            candidate_index: state.current ? state.current.videoCandidateIndex : null,
          });
          if (
            state.yt &&
            typeof state.yt.loadVideoById === "function" &&
            isYoutubeId(state.current.videoId)
          ) {
            state.yt.loadVideoById(state.current.videoId);
            state.yt.playVideo();
            setStatus("");
            return;
          }
        }
      }
    }
    if (
      state.yt &&
      typeof state.yt.loadVideoById === "function" &&
      state.current
    ) {
      postPlayerDebug("fallback_retry_current", {
        code: Number.isNaN(ytErrorCode) ? null : ytErrorCode,
        track: {
          title: state.current.title,
          artist: state.current.artist,
          videoId: state.current.videoId || "",
        },
        candidates: state.current.videoCandidateIds || [],
        candidate_index: state.current.videoCandidateIndex,
      });
      state.yt.loadVideoById(state.current.videoId);
      state.yt.playVideo();
      setStatus("");
      return;
    }
    if (await attemptSpotifyFallbackAfterYoutubeFailure(ytErrorCode)) {
      return;
    }
    const codeSuffix = Number.isNaN(ytErrorCode) ? "" : ` (YouTube error ${ytErrorCode})`;
    setStatus(`Cannot play this track right now${codeSuffix}.`, true);
    postPlayerDebug("fallback_giveup", {
      code: Number.isNaN(ytErrorCode) ? null : ytErrorCode,
      track: state.current
        ? {
            title: state.current.title,
            artist: state.current.artist,
            videoId: state.current.videoId || "",
          }
        : null,
      candidates: state.current ? state.current.videoCandidateIds || [] : [],
      candidate_index: state.current ? state.current.videoCandidateIndex : null,
    });
    } finally {
      state.fallbackInFlight = false;
    }
  }

  async function playTrackAt(index, listKey) {
    const key = listKey || state.activeListKey;
    const list = listFor(key);
    if (!Number.isInteger(index) || index < 0 || index >= list.length) return;

    state.activeListKey = key;
    state.queueIndex = index;
    if (key === "home") {
      const pageSize = Math.max(1, Number(state.homePaging.pageSize) || HOME_QUEUE_BATCH_SIZE);
      const targetPage = Math.floor(index / pageSize);
      if (Number.isInteger(targetPage) && targetPage >= 0) {
        state.homePaging.currentPage = targetPage;
        if (String(state.activeView || "") === "home") {
          renderHomeQueue();
        }
      }
    }
    if (state.shuffleEnabled) {
      const activeList = listFor(key);
      if (Array.isArray(activeList) && activeList.length) {
        getShufflePool(key, activeList.length, index);
      }
    }
    state.current = { ...list[index] };
    state.spotify.pendingAuthRetryTrackKey = "";
    state.spotify.pendingAuthRetryReasonCode = null;
    setPlaybackProvider("youtube");
    if (state.spotify.player && typeof state.spotify.player.pause === "function") {
      try {
        await state.spotify.player.pause();
      } catch (error) {}
    }
    state.current.embedBlockedAttempts = 0;
    state.current.lockedPlaybackVideoId =
      STRICT_METADATA_VIDEO_ID_PLAYBACK && isYoutubeId(state.current.videoId)
        ? String(state.current.videoId)
        : "";
    setCurrentVideoCandidates([state.current.videoId], state.current.videoId);
    state.pendingPlayRecordSongId = songIdForTrack(state.current);
    state.fallbackTried = false;
    state.fallbackInFlight = false;
    postPlayerDebug("track_click", {
      note: key,
      track: {
        title: state.current.title,
        artist: state.current.artist,
        videoId: state.current.videoId || "",
      },
      candidates: state.current.videoCandidateIds || [],
      candidate_index: state.current.videoCandidateIndex,
    });

    const playable =
      isYoutubeId(state.current.videoId) || (await resolveTrackSource(false));
    if (!playable) {
      setStatus("No playable source was found for this track.", true);
      return;
    }

    if ($.playerTitle) $.playerTitle.textContent = state.current.title;
    if ($.playerArtist) $.playerArtist.textContent = state.current.artist;
    if ($.player) $.player.classList.add("has-track");
    renderPlayerTrackMeta(state.current);
    setImgWithFallback(
      $.playerArt,
      state.current.coverCandidates,
      PLAYER_PLACEHOLDER,
      {
        track: state.current,
        allowYtDlpResolve: false,
      },
    );
    updateSelectedCard();

    if (state.yt && typeof state.yt.loadVideoById === "function") {
      postPlayerDebug("yt_load", {
        note: "initial_play",
        track: {
          title: state.current.title,
          artist: state.current.artist,
          videoId: state.current.videoId || "",
        },
        candidates: state.current.videoCandidateIds || [],
        candidate_index: state.current.videoCandidateIndex,
      });
      state.yt.loadVideoById(state.current.videoId);
      state.yt.playVideo();
      state.pendingStart = false;
      setStatus("");
    } else {
      state.pendingStart = true;
      setStatus("Player is loading...");
    }
    void hydrateCurrentTrackMeta();
    void hydrateLyrics(state.current);
  }

  function playPrevious() {
    const list = listFor(state.activeListKey);
    if (!Array.isArray(list) || !list.length) return;
    let targetIndex = -1;
    if (state.shuffleEnabled) {
      targetIndex = previousShuffleIndex(
        state.activeListKey,
        list.length,
        state.queueIndex,
      );
      if (!Number.isInteger(targetIndex) || targetIndex < 0) return;
    } else {
      if (state.queueIndex <= 0) return;
      targetIndex = state.queueIndex - 1;
    }
    if (state.current) void recordSkipEvent("prev", { ...state.current });
    void playTrackAt(targetIndex, state.activeListKey);
  }

  function playNext(isAutoAdvance) {
    const list = listFor(state.activeListKey);
    if (!list.length) return;
    if (state.queueIndex < 0) {
      if (!isAutoAdvance) void playTrackAt(0, state.activeListKey);
      return;
    }
    let nextIndex = -1;
    if (state.shuffleEnabled) {
      nextIndex = nextShuffleIndex(
        state.activeListKey,
        list.length,
        state.queueIndex,
      );
      if (!Number.isInteger(nextIndex) || nextIndex < 0) {
        nextIndex = state.queueIndex + 1;
      }
    } else {
      nextIndex = state.queueIndex + 1;
    }
    if (nextIndex >= list.length) {
      if (state.activeListKey === "home" && isAutoAdvance) {
        setHomeStatus("Loading more from your last Home query...");
        void appendMoreHomeQueueFromSource(true).then(function (appended) {
          if (appended) {
            setHomeStatus("");
            return;
          }
          const source = state.homeQueueSource;
          if (source && source.exhausted) {
            setHomeStatus("Home queue finished for the current query.");
          } else {
            setHomeStatus("Could not load more tracks for the current Home queue.", true);
          }
        });
      }
      return;
    }
    if (!isAutoAdvance && state.current) {
      void recordSkipEvent("next", { ...state.current });
    }
    void playTrackAt(nextIndex, state.activeListKey);
    if (state.activeListKey === "home") {
      void topUpHomeQueueByOneAfterAdvance();
    }
  }

  function togglePlay() {
    if (
      state.activePlaybackProvider === "spotify" &&
      state.spotify.player &&
      typeof state.spotify.player.togglePlay === "function"
    ) {
      try {
        void state.spotify.player.togglePlay();
      } catch (error) {}
      return;
    }
    if (!state.yt) return;
    if (state.isPlaying) state.yt.pauseVideo();
    else state.yt.playVideo();
  }

  function chooseEndpoint(query) {
    const q = query.toLowerCase();
    if (/\blike\s+[a-z0-9]/i.test(q)) return "/search";
    return MOOD_TERMS.some(function (word) {
      return q.includes(word);
    })
      ? "/sonic"
      : "/search";
  }

  function buildSearchCacheKey(endpoint, query, offset, limit) {
    return `${endpoint}|${String(query || "").toLowerCase()}|o=${Number(offset) || 0}|l=${Number(limit) || 0}`;
  }

  function mapSearchPayload(payload) {
    const seen = new Set();
    const tracks = [];
    (payload.results || []).forEach(function (raw) {
      const track = mapTrack(raw);
      const key =
        songIdForTrack(track) ||
        `${normalizeKeyPart(track.title)}-${normalizeKeyPart(track.artist)}`;
      if (seen.has(key)) return;
      seen.add(key);
      tracks.push(track);
    });
    return tracks;
  }

  function dedupeTracksForSearch(tracks) {
    const seen = new Set();
    const out = [];
    (Array.isArray(tracks) ? tracks : []).forEach(function (track) {
      if (!track) return;
      const key =
        songIdForTrack(track) ||
        `${normalizeKeyPart(track.title)}-${normalizeKeyPart(track.artist)}`;
      if (!key || seen.has(key)) return;
      seen.add(key);
      out.push(track);
    });
    return out;
  }

  function seedSearchPagingFromLocalTracks(query, queryId, tracks) {
    const pageSize = Math.max(
      1,
      Number(state.searchPaging.pageSize) || SEARCH_PAGE_SIZE,
    );
    const deduped = dedupeTracksForSearch(tracks);
    state.searchPaging.query = String(query || "");
    state.searchPaging.endpoint = "__local_fallback__";
    state.searchPaging.currentPage = 0;
    state.searchPaging.pages = new Map();
    state.searchPaging.pending = new Map();
    state.searchPaging.activeQueryId = Number(queryId) || 0;
    state.searchPaging.isPageLoading = false;
    for (
      let start = 0, page = 0;
      start < deduped.length;
      start += pageSize, page += 1
    ) {
      state.searchPaging.pages.set(page, deduped.slice(start, start + pageSize));
    }
    state.searchPaging.lastPageIndex =
      state.searchPaging.pages.size > 0
        ? state.searchPaging.pages.size - 1
        : null;
    return deduped;
  }

  function resetSearchPaging(query, endpoint, queryId) {
    state.searchPaging.query = String(query || "");
    state.searchPaging.endpoint = String(endpoint || "/search");
    state.searchPaging.pageSize = SEARCH_PAGE_SIZE;
    state.searchPaging.currentPage = 0;
    state.searchPaging.pages = new Map();
    state.searchPaging.pending = new Map();
    state.searchPaging.lastPageIndex = null;
    state.searchPaging.activeQueryId = Number(queryId) || 0;
    state.searchPaging.isPageLoading = false;
  }

  function updateSearchPagerUI() {
    if (!$.searchPager || !$.searchPrevBtn || !$.searchNextBtn) return;
    const paging = state.searchPaging;
    const currentTracks = paging.pages.get(paging.currentPage) || [];
    const hasResults = currentTracks.length > 0;
    const currentPageDisplay = Math.max(1, Number(paging.currentPage || 0) + 1);
    const totalPages =
      paging.lastPageIndex !== null
        ? Math.max(1, Number(paging.lastPageIndex || 0) + 1)
        : null;

    $.searchPager.classList.toggle("hidden", !hasResults);
    if ($.searchPageLabel) {
      $.searchPageLabel.textContent = hasResults
        ? totalPages
          ? `Page ${currentPageDisplay} of ${totalPages}`
          : `Page ${currentPageDisplay}`
        : "";
    }
    if (!hasResults) {
      $.searchPrevBtn.disabled = true;
      $.searchNextBtn.disabled = true;
      $.searchNextBtn.textContent = "Next 15";
      return;
    }

    const nextPageIndex = paging.currentPage + 1;
    const nextCached = paging.pages.has(nextPageIndex);
    const nextPending = paging.pending.has(nextPageIndex);
    const nextTracks = nextCached ? paging.pages.get(nextPageIndex) || [] : [];
    const reachedEnd =
      paging.lastPageIndex !== null &&
      paging.currentPage >= paging.lastPageIndex;
    const loadingNextOnly = nextPending && !nextCached;

    $.searchPrevBtn.disabled = paging.isPageLoading || paging.currentPage <= 0;
    $.searchNextBtn.disabled =
      paging.isPageLoading ||
      reachedEnd ||
      loadingNextOnly ||
      (nextCached && nextTracks.length <= 0);
    $.searchNextBtn.textContent = loadingNextOnly
      ? "Loading next..."
      : "Next 15";
  }

  async function fetchSmartSearchFallbackTracks(query, activeEndpoint, signal) {
    const q = String(query || "").trim();
    if (!q) {
      return { mode: "none", endpoint: "", tracks: [], message: "" };
    }

    const primary = String(activeEndpoint || "").trim() || "/search";
    const alternateEndpoints = ["/search", "/sonic"].filter(function (ep) {
      return ep !== primary;
    });
    for (const endpoint of alternateEndpoints) {
      try {
        const payload = await fetchSearchPagePayload(
          q,
          endpoint,
          0,
          SEARCH_PAGE_SIZE,
          signal,
          { timeoutMs: 10000, retries: 1 },
        );
        const tracks = mapSearchPayload(payload);
        if (tracks.length) {
          return {
            mode: "endpoint",
            endpoint,
            tracks,
            message: "No exact match. Showing closest results.",
          };
        }
      } catch (error) {}
    }

    const merged = [];
    const fallbackPoolSize = Math.max(96, SEARCH_PAGE_SIZE * 8);
    if (state.userId) {
      try {
        const payload = await requestJSON(
          `/api/recommend?user_id=${encodeURIComponent(state.userId)}&n=${fallbackPoolSize}&covers=0`,
          signal ? { signal } : {},
          { timeoutMs: 13000, retries: 0 },
        );
        const recommended = Array.isArray(payload && payload.recommendations)
          ? payload.recommendations.map(mapRecommendation)
          : [];
        merged.push(...recommended);
      } catch (error) {}
    }
    try {
      const trending = await fetchTrendingFallback(fallbackPoolSize, signal);
      merged.push(...trending);
    } catch (error) {}

    if (
      !merged.length &&
      Array.isArray(state.lists.foryou) &&
      state.lists.foryou.length
    ) {
      merged.push(...state.lists.foryou);
    }
    if (
      !merged.length &&
      Array.isArray(state.lists.home) &&
      state.lists.home.length
    ) {
      merged.push(...state.lists.home);
    }

    const deduped = dedupeTracksForSearch(merged);
    if (!deduped.length) {
      return { mode: "none", endpoint: "", tracks: [], message: "" };
    }
    return {
      mode: "local",
      endpoint: "__local_fallback__",
      tracks: deduped,
      message: "No exact match. Showing smart picks you can browse.",
    };
  }

  async function fetchSearchPagePayload(
    query,
    endpoint,
    offset,
    limit,
    signal,
    requestCfg,
  ) {
    const q = String(query || "").trim();
    const normalizedOffset = Math.max(0, Number(offset) || 0);
    const normalizedLimit = Math.max(1, Number(limit) || SEARCH_PAGE_SIZE);
    const cfg = requestCfg || {};
    const cacheKey = buildSearchCacheKey(
      endpoint,
      q,
      normalizedOffset,
      normalizedLimit,
    );
    let payload = state.searchCache.get(cacheKey);
    if (!payload) {
      payload = await requestJSON(
        `${endpoint}?q=${encodeURIComponent(q)}&limit=${normalizedLimit}&offset=${normalizedOffset}`,
        signal ? { signal } : {},
        {
          timeoutMs: Number(cfg.timeoutMs || 9200),
          retries: Number.isFinite(Number(cfg.retries)) ? Number(cfg.retries) : 1,
        },
      );
      if (state.searchCache.size >= 60) {
        const oldest = state.searchCache.keys().next().value;
        state.searchCache.delete(oldest);
      }
      state.searchCache.set(cacheKey, payload);
    }
    return payload;
  }

  async function ensureSearchPage(pageIndex, signal) {
    const paging = state.searchPaging;
    const nextPage = Number(pageIndex);
    if (!Number.isInteger(nextPage) || nextPage < 0) return [];
    if (paging.lastPageIndex !== null && nextPage > paging.lastPageIndex)
      return [];

    if (paging.pages.has(nextPage)) {
      return paging.pages.get(nextPage) || [];
    }
    if (paging.pending.has(nextPage)) {
      return paging.pending.get(nextPage);
    }

    const expectedQueryId = paging.activeQueryId;
    const offset = nextPage * paging.pageSize;
    const promise = (async function () {
      const payload = await fetchSearchPagePayload(
        paging.query,
        paging.endpoint,
        offset,
        paging.pageSize,
        signal,
      );
      const tracks = mapSearchPayload(payload);
      if (tracks.length) {
        await Promise.allSettled(
          tracks.slice(0, paging.pageSize).map(function (track) {
            if (trackHasNonYoutubeCover(track)) return Promise.resolve(track);
            return preResolveTrackCoverIfNeededBounded(
              track,
              FOR_YOU_PRERESOLVE_TIMEOUT_MS,
            );
          }),
        );
      }
      if (state.searchPaging.activeQueryId !== expectedQueryId) return [];
      state.searchPaging.pages.set(nextPage, tracks);
      if (tracks.length < paging.pageSize) {
        state.searchPaging.lastPageIndex = nextPage;
      }
      return tracks;
    })();
    paging.pending.set(nextPage, promise);

    try {
      return await promise;
    } finally {
      paging.pending.delete(nextPage);
    }
  }

  async function prefetchSearch(query) {
    const q = String(query || "").trim();
    if (q.length < 3) return;
    const endpoint = chooseEndpoint(q);
    try {
      await fetchSearchPagePayload(q, endpoint, 0, SEARCH_PAGE_SIZE, null);
    } catch (error) {
      return;
    }
  }

  async function prefetchNextSearchPage(pageIndex) {
    const paging = state.searchPaging;
    const targetPage = Number(pageIndex);
    if (!Number.isInteger(targetPage) || targetPage < 0) return;
    if (paging.pages.has(targetPage) || paging.pending.has(targetPage)) return;
    if (paging.lastPageIndex !== null && targetPage > paging.lastPageIndex)
      return;
    const expectedQueryId = paging.activeQueryId;
    const prefetchPromise = ensureSearchPage(targetPage, null);
    updateSearchPagerUI();
    try {
      await prefetchPromise;
    } catch (error) {
      return;
    } finally {
      if (state.searchPaging.activeQueryId === expectedQueryId) {
        updateSearchPagerUI();
      }
    }
  }

  function applySearchPage(pageIndex) {
    const pageTracks = state.searchPaging.pages.get(pageIndex) || [];
    state.searchPaging.currentPage = pageIndex;
    state.activeListKey = "search";
    state.lists.search = pageTracks.slice();
    state.queueIndex = -1;
    void prefetchNextSearchPage(pageIndex + 1);
    renderSearchResults();
    updateSearchPagerUI();
  }

  async function goToSearchPage(pageIndex) {
    const paging = state.searchPaging;
    const targetPage = Number(pageIndex);
    if (!Number.isInteger(targetPage) || targetPage < 0) return;
    if (targetPage === paging.currentPage) return;
    if (paging.lastPageIndex !== null && targetPage > paging.lastPageIndex)
      return;
    const expectedQueryId = paging.activeQueryId;

    paging.isPageLoading = true;
    updateSearchPagerUI();
    setStatus("Loading tracks...");

    try {
      const tracks = await ensureSearchPage(
        targetPage,
        state.searching ? state.searching.signal : null,
      );
      if (state.searchPaging.activeQueryId !== expectedQueryId) return;
      if (!tracks.length) {
        setStatus("");
        updateSearchPagerUI();
        return;
      }
      applySearchPage(targetPage);
      setStatus("");
    } catch (error) {
      if (error.name === "AbortError") return;
      setStatus("Could not load more tracks.", true);
    } finally {
      paging.isPageLoading = false;
      updateSearchPagerUI();
    }
  }

  async function search() {
    const query = String($.query && $.query.value ? $.query.value : "").trim();
    if (!query) return;
    switchView("search");
    const endpoint = chooseEndpoint(query);

    state.queryId += 1;
    const id = state.queryId;

    if (state.searching) state.searching.abort();
    state.searching = new AbortController();

    setStatus("Searching...");
    if ($.results) $.results.innerHTML = "";
    resetSearchPaging(query, endpoint, id);
    updateSearchPagerUI();

    try {
      let tracks = await ensureSearchPage(0, state.searching.signal);
      let fallbackStatus = "";
      if (id !== state.queryId) return;

      if (!tracks.length) {
        const fallback = await fetchSmartSearchFallbackTracks(
          query,
          state.searchPaging.endpoint,
          state.searching.signal,
        );
        if (id !== state.queryId) return;
        if (fallback.mode === "endpoint" && fallback.endpoint) {
          resetSearchPaging(query, fallback.endpoint, id);
          updateSearchPagerUI();
          tracks = await ensureSearchPage(0, state.searching.signal);
          if (id !== state.queryId) return;
          fallbackStatus = String(fallback.message || "");
        } else if (fallback.mode === "local" && fallback.tracks.length) {
          seedSearchPagingFromLocalTracks(query, id, fallback.tracks);
          applySearchPage(0);
          setStatus(String(fallback.message || ""));
          return;
        }
      }

      if (!tracks.length) {
        state.lists.search = [];
        state.queueIndex = -1;
        renderEmpty($.results, "Could not load related tracks right now.");
        updateSearchPagerUI();
        setStatus("Search fallback exhausted. Try another prompt.", true);
        return;
      }

      applySearchPage(0);
      setStatus(fallbackStatus || "");
    } catch (error) {
      if (error.name === "AbortError") return;
      state.lists.search = [];
      state.queueIndex = -1;
      renderEmpty($.results, "Connection error.");
      updateSearchPagerUI();
      setStatus("Search request failed. Try again.", true);
    }
  }

  async function setTrackLiked(songId, nextLiked) {
    const id = String(songId || "").trim();
    if (!id || !state.userId) return;
    if (state.pendingLikeSongIds.has(id) || state.pendingDislikeSongIds.has(id)) return;

    const wasLiked = state.likedSongIds.has(id);
    if (wasLiked === nextLiked) return;

    state.pendingLikeSongIds.add(id);
    if (nextLiked) state.likedSongIds.add(id);
    else state.likedSongIds.delete(id);
    syncLikeUI();

    try {
      await requestJSON(
        nextLiked ? "/api/like" : "/api/unlike",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            song_id: id,
          }),
        },
        { timeoutMs: 5200, retries: 1 },
      );
      void loadUserProfile(true);
      if (nextLiked) {
        // If the song was liked, it should not stay in any explicit negative set locally.
        state.dislikedSongIds.delete(id);
      }
      if (state.activeView === "liked") {
        void loadLikedSongs(true);
      } else {
        // Force refresh next time the user opens Liked Songs.
        state.lists.liked = [];
      }
      scheduleForYouRefresh(700);
    } catch (error) {
      if (wasLiked) state.likedSongIds.add(id);
      else state.likedSongIds.delete(id);
      syncLikeUI();
      setStatus("Could not update like right now.", true);
    } finally {
      state.pendingLikeSongIds.delete(id);
      syncLikeUI();
    }
  }

  function toggleLikeSong(songId) {
    const id = String(songId || "").trim();
    if (!id) return Promise.resolve();
    return setTrackLiked(id, !state.likedSongIds.has(id));
  }

  async function setTrackDisliked(songId, nextDisliked) {
    const id = String(songId || "").trim();
    if (!id || !state.userId) return;
    if (state.pendingDislikeSongIds.has(id) || state.pendingLikeSongIds.has(id)) return;

    const wasDisliked = state.dislikedSongIds.has(id);
    if (wasDisliked === nextDisliked) return;

    state.pendingDislikeSongIds.add(id);
    if (nextDisliked) state.dislikedSongIds.add(id);
    else state.dislikedSongIds.delete(id);
    if (nextDisliked) state.likedSongIds.delete(id);
    syncLikeUI();

    try {
      await requestJSON(
        nextDisliked ? "/api/dislike" : "/api/undislike",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            song_id: id,
          }),
        },
        { timeoutMs: 5200, retries: 1 },
      );
      void loadUserProfile(true);
      if (state.activeView === "liked") {
        void loadLikedSongs(true);
      } else {
        state.lists.liked = [];
      }
      scheduleForYouRefresh(700);
    } catch (error) {
      if (wasDisliked) state.dislikedSongIds.add(id);
      else state.dislikedSongIds.delete(id);
      syncLikeUI();
      setStatus("Could not update dislike right now.", true);
    } finally {
      state.pendingDislikeSongIds.delete(id);
      syncLikeUI();
    }
  }

  function toggleDislikeSong(songId) {
    const id = String(songId || "").trim();
    if (!id) return Promise.resolve();
    return setTrackDisliked(id, !state.dislikedSongIds.has(id));
  }

  function scheduleForYouRefresh(delayMs) {
    if (state.feedRefreshTimer) clearTimeout(state.feedRefreshTimer);
    state.feedRefreshTimer = setTimeout(
      function () {
        void loadForYou(true);
      },
      Number(delayMs || 800),
    );
  }

  async function recordCurrentPlayOnce() {
    if (!state.userId) return;
    const songId = String(state.pendingPlayRecordSongId || "").trim();
    if (!songId) return;
    const currentTrack = state.current ? { ...state.current } : null;
    const lyricsKey = lyricsCacheKeyForTrack(currentTrack);
    const playSource = String(state.activeListKey || "").trim() || "search";
    const activeQuery =
      playSource === "search"
        ? String(state.searchPaging.query || ($.query && $.query.value) || "").trim()
        : "";
    state.pendingPlayRecordSongId = "";
    try {
      await requestJSON(
        "/api/play",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: state.userId,
            song_id: songId,
            title: currentTrack && currentTrack.title ? currentTrack.title : "",
            artist: currentTrack && currentTrack.artist ? currentTrack.artist : "",
            source: playSource,
            query: activeQuery,
          }),
        },
        { timeoutMs: 4500, retries: 1 },
      );
      if (lyricsKey) state.lyricsPrefetchDoneAt.delete(lyricsKey);
      state.profile.interaction_count =
        Number(state.profile.interaction_count || 0) + 1;
      updatePersonalSummary();
      if (state.activeView === "history") {
        void loadHistoryTracks(true);
      } else {
        state.lists.history = [];
      }
    } catch (error) {
      return;
    }
  }

  async function recordSkipEvent(eventKind, trackSnapshot) {
    const event = String(eventKind || "").trim().toLowerCase();
    if (!state.userId || (event !== "next" && event !== "prev")) return;
    const track = trackSnapshot && typeof trackSnapshot === "object" ? trackSnapshot : state.current;
    const songId = String(songIdForTrack(track) || "").trim();
    if (!songId) return;
    const timing = getActivePlaybackTiming();
    const payload = {
      user_id: state.userId,
      song_id: songId,
      event,
      position_sec: Number(timing && timing.current ? timing.current : 0),
      duration_sec: Number(timing && timing.duration ? timing.duration : 0),
    };
    try {
      await requestJSON(
        "/api/skip",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        { timeoutMs: 4200, retries: 0 },
      );
      void loadUserProfile(true);
      scheduleForYouRefresh(550);
    } catch (error) {}
  }

  async function loadUserProfile(isSilent) {
    if (!state.userId) return;
    try {
      const profile = await requestJSON(
        `/api/profile?user_id=${encodeURIComponent(state.userId)}`,
        {},
        { timeoutMs: 6500, retries: 1 },
      );
      state.profile = profile || {};
      state.likedSongIds = new Set(
        (Array.isArray(profile.likes) ? profile.likes : []).map(function (id) {
          return String(id);
        }),
      );
      state.dislikedSongIds = new Set(
        (Array.isArray(profile.dislikes) ? profile.dislikes : []).map(function (id) {
          return String(id);
        }),
      );
      updatePersonalSummary();
      syncLikeUI();
    } catch (error) {
      if (!isSilent) {
        setForYouStatus(
          "Profile load failed. Continuing in fallback mode.",
          true,
        );
      }
    }
  }

  function mapProfileRows(rows) {
    const seen = new Set();
    const tracks = [];
    (Array.isArray(rows) ? rows : []).forEach(function (raw) {
      const track = mapRecommendation(raw || {});
      const key =
        songIdForTrack(track) ||
        `${normalizeKeyPart(track.title)}-${normalizeKeyPart(track.artist)}`;
      if (!key || seen.has(key)) return;
      seen.add(key);
      tracks.push(track);
    });
    return tracks;
  }

  async function loadLikedSongs(isSilent) {
    if (!state.userId) return;
    if (!isSilent) setLikedStatus("Loading liked songs...");
    try {
      const payload = await requestJSON(
        `/api/liked?user_id=${encodeURIComponent(state.userId)}&n=${PROFILE_LIBRARY_FETCH_SIZE}&covers=0`,
        {},
        { timeoutMs: 10000, retries: 0 },
      );
      const tracks = mapProfileRows(payload && payload.liked);
      await Promise.allSettled(
        tracks.slice(0, 10).map(function (track) {
          if (trackHasNonYoutubeCover(track)) return Promise.resolve(track);
          return preResolveTrackCoverIfNeededBounded(track, FOR_YOU_PRERESOLVE_TIMEOUT_MS);
        }),
      );
      state.lists.liked = tracks;
      renderLikedSongs();
      setLikedStatus("");
    } catch (error) {
      state.lists.liked = [];
      renderLikedSongs();
      setLikedStatus("Could not load liked songs.", true);
    }
  }

  async function loadHistoryTracks(isSilent) {
    if (!state.userId) return;
    if (!isSilent) setHistoryStatus("Loading history...");
    try {
      const payload = await requestJSON(
        `/api/history?user_id=${encodeURIComponent(state.userId)}&n=${PROFILE_LIBRARY_FETCH_SIZE}&covers=0`,
        {},
        { timeoutMs: 10000, retries: 0 },
      );
      const tracks = mapProfileRows(payload && payload.history);
      await Promise.allSettled(
        tracks.slice(0, 10).map(function (track) {
          if (trackHasNonYoutubeCover(track)) return Promise.resolve(track);
          return preResolveTrackCoverIfNeededBounded(track, FOR_YOU_PRERESOLVE_TIMEOUT_MS);
        }),
      );
      state.lists.history = tracks;
      renderHistoryTracks();
      setHistoryStatus("");
    } catch (error) {
      state.lists.history = [];
      renderHistoryTracks();
      setHistoryStatus("Could not load history.", true);
    }
  }

  async function fetchTrendingFallback(requestedN, signal) {
    const n = Math.max(24, Number(requestedN) || FOR_YOU_FETCH_POOL_SIZE);
    const payload = await requestJSON(
      `/api/trending?n=${n}&covers=0`,
      signal ? { signal } : {},
      { timeoutMs: 12000, retries: 0 },
    );
    const tracks = Array.isArray(payload.trending) ? payload.trending : [];
    return tracks.map(mapRecommendation);
  }

  function mergeUniqueTracks(baseTracks, extraTracks) {
    const merged = [];
    const seen = new Set();
    const pushUnique = function (track) {
      const id =
        forYouTrackStableId(track) ||
        String(songIdForTrack(track) || "").trim() ||
        `${normalizeKeyPart(track && track.title)}-${normalizeKeyPart(track && track.artist)}`;
      if (!id || seen.has(id)) return;
      seen.add(id);
      merged.push(track);
    };
    (Array.isArray(baseTracks) ? baseTracks : []).forEach(pushUnique);
    (Array.isArray(extraTracks) ? extraTracks : []).forEach(pushUnique);
    return merged;
  }

  async function ensureForYouPoolMinCount(tracks, minCount, options) {
    const opts = options || {};
    const allowTrendingFallback = opts.allowTrendingFallback !== false;
    const target = Math.max(1, Number(minCount) || 1);
    let merged = mergeUniqueTracks(tracks, []);
    if (merged.length >= target) return merged;
    if (!allowTrendingFallback) return merged;
    try {
      const fallback = await fetchTrendingFallback(
        Math.max(target * 2, FOR_YOU_FETCH_POOL_SIZE),
      );
      merged = mergeUniqueTracks(merged, fallback);
    } catch (error) {
      // Keep personalized-only list if fallback cannot be fetched.
    }
    return merged;
  }

  function forYouSignalSnapshot(historyCounts) {
    const counts =
      historyCounts && typeof historyCounts === "object" ? historyCounts : {};
    const likesFromProfile = Array.isArray(state.profile && state.profile.likes)
      ? state.profile.likes.length
      : Number(state.likedSongIds && state.likedSongIds.size ? state.likedSongIds.size : 0);
    const playsFromProfile = Array.isArray(state.profile && state.profile.recent_plays)
      ? state.profile.recent_plays.length
      : 0;
    const playlistsFromProfile = Number(
      state.profile && state.profile.playlist_track_count
        ? state.profile.playlist_track_count
        : 0,
    );
    const likes = Math.max(0, Number(counts.likes || likesFromProfile || 0));
    const plays = Math.max(0, Number(counts.plays || playsFromProfile || 0));
    const playlistTracks = Math.max(
      0,
      Number(counts.playlist_tracks || playlistsFromProfile || 0),
    );
    return { likes, plays, playlistTracks };
  }

  function hasPersonalSignalsForYou(historyCounts) {
    const snapshot = forYouSignalSnapshot(historyCounts);
    return (
      Number(snapshot.likes || 0) > 0 ||
      Number(snapshot.plays || 0) > 0 ||
      Number(snapshot.playlistTracks || 0) > 0
    );
  }

  function scheduleForYouDeepRefresh(delayMs) {
    if (!state.userId) return;
    if (state.forYouDeepRefreshTimer) {
      clearTimeout(state.forYouDeepRefreshTimer);
      state.forYouDeepRefreshTimer = null;
    }
    const waitMs = Math.max(120, Number(delayMs) || 260);
    state.forYouDeepRefreshTimer = setTimeout(function () {
      state.forYouDeepRefreshTimer = null;
      if (state.forYouPaging.loading) return;
      const now = Date.now();
      if (now - Number(state.forYouDeepRefreshLastAt || 0) < 10_000) return;
      void loadForYou(true, {
        forceDeep: true,
        backgroundRefresh: true,
        keepPage: true,
      });
    }, waitMs);
  }

  async function loadForYou(isSilent, options) {
    if (!state.userId) return;
    if (state.forYouPaging.loading) return;
    const opts = options || {};
    const appendPool = Boolean(opts.appendPool);
    const initialLoad = Boolean(opts.initialLoad);
    const backgroundRefresh = Boolean(opts.backgroundRefresh);
    const keepPage = Boolean(opts.keepPage);
    const fastMode = opts.forceDeep
      ? false
      : (opts.fastMode != null ? Boolean(opts.fastMode) : true);
    const existingPool = Array.isArray(state.forYouPaging.tracks)
      ? state.forYouPaging.tracks.slice()
      : [];
    const basePool = appendPool
      ? existingPool.slice()
      : [];
    state.forYouPaging.loading = true;
    if (!isSilent && !backgroundRefresh) {
      setForYouStatus(appendPool ? "Loading more picks..." : "Loading quick personalized picks...");
    }
    updateForYouPageSize(true);
    const pageSize = Math.max(1, Number(state.forYouPaging.pageSize) || FOR_YOU_PAGE_SIZE);
    const minPagedTracks = appendPool
      ? Math.max(basePool.length + pageSize * 3, pageSize * 6)
      : Math.max(pageSize * 6, FOR_YOU_FETCH_POOL_SIZE);
    const requestPoolSize = fastMode
      ? Math.min(
          180,
          Math.max(pageSize * (appendPool ? 3 : 2), 96),
        )
      : Math.min(
          360,
          Math.max(FOR_YOU_FETCH_POOL_SIZE, minPagedTracks),
        );
    const requestTimeoutMs = fastMode
      ? (initialLoad ? 9000 : 7000)
      : (initialLoad ? 18000 : 15000);
    const requestRetries = fastMode ? 0 : (initialLoad ? 1 : 0);
    if (!state.lists.foryou.length && !existingPool.length && !backgroundRefresh) {
      renderForYouSkeleton(Math.min(pageSize, 12));
    }
    const currentBatchIds = getForYouPageTracks(state.forYouPaging.currentPage)
      .map(function (track) {
        return forYouTrackStableId(track);
      })
      .filter(Boolean);

    try {
      const payload = await requestJSON(
        `/api/recommend?user_id=${encodeURIComponent(state.userId)}&n=${requestPoolSize}&covers=0&fast=${fastMode ? "1" : "0"}`,
        {},
        { timeoutMs: requestTimeoutMs, retries: requestRetries },
      );
      const recommendationMode = String(payload && payload.recommendation_mode ? payload.recommendation_mode : "");
      const historyCounts =
        payload && payload.history_counts && typeof payload.history_counts === "object"
          ? payload.history_counts
          : {};
      const signalSnapshot = forYouSignalSnapshot(historyCounts);
      const likesCount = signalSnapshot.likes;
      const playsCount = signalSnapshot.plays;
      const playlistCount = signalSnapshot.playlistTracks;
      const hasPersonalSignals = hasPersonalSignalsForYou(historyCounts);
      let tracks = Array.isArray(payload.recommendations)
        ? payload.recommendations.map(mapRecommendation)
        : [];
      tracks = shuffleTracksCopy(tracks);
      if (opts.avoidCurrentBatch && currentBatchIds.length) {
        const avoidSet = new Set(currentBatchIds);
        const preferred = [];
        const deferred = [];
        tracks.forEach(function (track) {
          const id = forYouTrackStableId(track);
          if (id && avoidSet.has(id)) deferred.push(track);
          else preferred.push(track);
        });
        tracks = preferred.concat(deferred);
      }
      tracks = await ensureForYouPoolMinCount(tracks, minPagedTracks, {
        allowTrendingFallback: false,
      });
      const mergedTracks = appendPool
        ? mergeUniqueTracks(basePool, tracks)
        : mergeUniqueTracks(tracks, existingPool);
      state.forYouPaging.exhausted = appendPool && mergedTracks.length <= basePool.length;
      setForYouTrackPool(mergedTracks, { keepPage: appendPool || keepPage || backgroundRefresh });
      await preResolveForYouPageCovers(
        (appendPool || keepPage || backgroundRefresh) ? state.forYouPaging.currentPage : 0,
      );
      renderForYou();
      if (state.forYouPaging.exhausted) {
        setForYouStatus("No more unique recommendations right now. Try Refresh All for a new mix.");
      } else if (recommendationMode === "adaptive") {
        setForYouStatus(
          `Deep personalized from your profile (${likesCount} likes, ${playsCount} plays, ${playlistCount} playlist tracks).`,
        );
        state.forYouDeepRefreshLastAt = Date.now();
      } else if (
        recommendationMode === "personalized_seed" ||
        recommendationMode === "profile_seed_fast"
      ) {
        if (hasPersonalSignals) {
          setForYouStatus(
            `Quick personalized picks ready (${likesCount} likes, ${playsCount} plays, ${playlistCount} playlist tracks). Deep reranking is refreshing in background.`,
          );
          if (!appendPool) {
            scheduleForYouDeepRefresh(180);
          }
        } else {
          setForYouStatus("Personalized feed is initializing from your profile.");
        }
      } else if (
        recommendationMode === "warming_up"
      ) {
        if (existingPool.length) {
          setForYouStatus("Quick personalized picks are ready. Deep personalization is still warming up.");
        } else {
          setForYouStatus("Deep personalization is warming up. Your profile-based feed will appear shortly.");
        }
      } else if (
        recommendationMode === "cold_start_fast" ||
        recommendationMode === "cold_start"
      ) {
        if (hasPersonalSignals) {
          setForYouStatus("Deep personalization is initializing. Tap Refresh All to retry.");
        } else {
          setForYouStatus("Like and play tracks to unlock personalized recommendations.");
        }
      } else {
        setForYouStatus("");
      }
    } catch (error) {
      if (appendPool) {
        state.forYouPaging.exhausted = true;
        renderForYou();
        setForYouStatus("Could not load more personalized picks right now.", true);
      } else if (existingPool.length) {
        setForYouTrackPool(existingPool, { keepPage: true });
        await preResolveForYouPageCovers(state.forYouPaging.currentPage);
        renderForYou();
        setForYouStatus("Keeping your current personalized feed while deep personalization retries.");
      } else {
        setForYouTrackPool([], { keepPage: false });
        state.forYouPaging.exhausted = true;
        state.lists.foryou = [];
        renderForYou();
        setForYouStatus("Deep personalization is warming up. Your recommendations will appear shortly.");
      }
    } finally {
      state.forYouPaging.loading = false;
      if (opts.forceDeep) {
        state.forYouDeepRefreshLastAt = Date.now();
      }
    }
  }

  function onPlayerStateChange(event) {
    if (!window.YT || !window.YT.PlayerState) return;
    if (
      state.activePlaybackProvider === "spotify" &&
      event.data !== window.YT.PlayerState.PLAYING
    ) {
      return;
    }

    if (event.data === window.YT.PlayerState.PLAYING) {
      if (state.current) state.current.embedBlockedAttempts = 0;
      setPlaybackProvider("youtube");
      postPlayerDebug("yt_state_playing", {
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId || "",
            }
          : null,
        candidates: state.current ? state.current.videoCandidateIds || [] : [],
        candidate_index: state.current ? state.current.videoCandidateIndex : null,
      });
      setPlayState(true);
      startProgress();
      updateProgress(true);
      void recordCurrentPlayOnce();
      return;
    }

    if (event.data === window.YT.PlayerState.ENDED) {
      postPlayerDebug("yt_state_ended", {
        track: state.current
          ? {
              title: state.current.title,
              artist: state.current.artist,
              videoId: state.current.videoId || "",
            }
          : null,
      });
      setPlayState(false);
      stopProgress();
      updateProgress(true);
      if (state.repeatOne) {
        void replayCurrentTrackAfterEnd("youtube");
        return;
      }
      playNext(true);
      return;
    }

    if (event.data === window.YT.PlayerState.BUFFERING) return;

    setPlayState(false);
    stopProgress();
    updateProgress(true);
  }

  function initYouTubePlayer() {
    const applyYouTubeIframeReferrerPolicy = function (retriesLeft) {
      let iframe = null;
      try {
        if (state.yt && typeof state.yt.getIframe === "function") {
          iframe = state.yt.getIframe();
        }
      } catch (error) {}
      if (!iframe) {
        iframe =
          document.querySelector("iframe#hidden-yt") ||
          document.querySelector("#hidden-yt iframe");
      }
      if (iframe) {
        const policy = "strict-origin-when-cross-origin";
        iframe.referrerPolicy = policy;
        iframe.setAttribute(
          "referrerpolicy",
          policy,
        );
        try {
          const allowRaw = String(iframe.getAttribute("allow") || "").trim();
          const allowParts = allowRaw
            ? allowRaw
                .split(/[;,]/)
                .map(function (part) {
                  return String(part || "").trim();
                })
                .filter(Boolean)
            : [];
          const allowSet = new Set(allowParts.map(function (part) { return part.toLowerCase(); }));
          if (!allowSet.has("autoplay")) allowParts.push("autoplay");
          if (!allowSet.has("encrypted-media")) allowParts.push("encrypted-media");
          iframe.setAttribute("allow", allowParts.join("; "));
        } catch (error) {}

        // Force privacy-enhanced embed host + embed path on the generated iframe URL.
        // Keep JS API params intact so the YT Iframe API still controls the player.
        try {
          const rawSrc = String(iframe.getAttribute("src") || iframe.src || "").trim();
          if (rawSrc) {
            const url = new URL(rawSrc, window.location.href);
            const host = String(url.hostname || "").toLowerCase();
            const isYouTubeHost =
              host === "www.youtube.com" ||
              host === "youtube.com" ||
              host === "m.youtube.com" ||
              host === "www.youtube-nocookie.com" ||
              host === "youtube-nocookie.com";
            if (isYouTubeHost) {
              if ((url.pathname || "").toLowerCase() === "/watch") {
                const watchId = String(url.searchParams.get("v") || "").trim();
                if (watchId) {
                  url.pathname = `/embed/${watchId}`;
                  url.searchParams.delete("v");
                }
              }
              url.hostname = "www.youtube-nocookie.com";
              if (window.location && window.location.origin) {
                if (!url.searchParams.get("enablejsapi")) {
                  url.searchParams.set("enablejsapi", "1");
                }
                if (!url.searchParams.get("origin")) {
                  url.searchParams.set("origin", window.location.origin);
                }
              }
              const nextSrc = url.toString();
              if (nextSrc && nextSrc !== rawSrc) {
                iframe.src = nextSrc;
              }
            }
          }
        } catch (error) {}
        return true;
      }
      if ((Number(retriesLeft) || 0) > 0) {
        setTimeout(function () {
          applyYouTubeIframeReferrerPolicy(Number(retriesLeft) - 1);
        }, 120);
      }
      return false;
    };

    state.yt = new window.YT.Player("hidden-yt", {
      host: "https://www.youtube-nocookie.com",
      height: "1",
      width: "1",
      playerVars: {
        playsinline: 1,
        autoplay: 1,
        controls: 0,
        disablekb: 1,
        fs: 0,
        origin: window.location.origin,
      },
      events: {
        onReady: function () {
          applyYouTubeIframeReferrerPolicy(25);
        },
        onStateChange: onPlayerStateChange,
        onError: function (event) {
          void resolveFallback(event);
        },
      },
    });

    setVolume($.volume ? $.volume.value : 100);
    if (state.current && isYoutubeId(state.current.videoId)) {
      state.yt.loadVideoById(state.current.videoId);
      if (state.pendingStart) {
        state.yt.playVideo();
        state.pendingStart = false;
      }
    }
    applyYouTubeIframeReferrerPolicy(25);
    setStatus("");
  }

  function bindEvents() {
    $.navItems.forEach(function (item) {
      item.addEventListener("click", function (event) {
        event.preventDefault();
        const view = String(item.dataset.view || "");
        if (!view) return;
        switchView(view);
        if (view === "search" && $.query) $.query.focus();
        if (view === "home" && $.homeQuery) $.homeQuery.focus();
        if (view === "foryou" && !state.lists.foryou.length) void loadForYou(false);
        if (view === "liked") void loadLikedSongs(false);
        if (view === "history") void loadHistoryTracks(false);
        if (view === "playlists") void loadPlaylists(false);
      });
    });

    $.settingsButtons.forEach(function (button) {
      if (!button) return;
      button.setAttribute("aria-haspopup", "menu");
      button.setAttribute("aria-expanded", "false");
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        toggleSettingsMenu(button);
      });
    });

    if ($.settingsSupportLink) {
      $.settingsSupportLink.addEventListener("click", function () {
        closeSettingsMenu();
      });
    }

    document.addEventListener("click", function (event) {
      if (!isSettingsMenuOpen()) return;
      const target = event && event.target ? event.target : null;
      const clickedInsideMenu =
        target && typeof target.closest === "function"
          ? target.closest("#settings-menu")
          : null;
      const clickedSettingsButton =
        target && typeof target.closest === "function"
          ? target.closest('.icon-btn[aria-label="Settings"]')
          : null;
      if (clickedInsideMenu || clickedSettingsButton) return;
      closeSettingsMenu();
    });

    document.addEventListener(
      "scroll",
      function () {
        if (isSettingsMenuOpen()) closeSettingsMenu();
      },
      true,
    );

    window.addEventListener("resize", function () {
      if (!isSettingsMenuOpen()) return;
      positionSettingsMenu(state.settingsMenu.anchorButton);
    });

    $.userAvatars.forEach(function (avatar) {
      if (!avatar) return;
      avatar.setAttribute("role", "button");
      avatar.setAttribute("tabindex", "0");
      avatar.addEventListener("click", function () {
        openAuthModal();
      });
      avatar.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openAuthModal();
        }
      });
    });

    if ($.authModalClose) {
      $.authModalClose.addEventListener("click", function () {
        closeAuthModal();
      });
    }
    if ($.authModal) {
      $.authModal.addEventListener("click", function (event) {
        const closeTarget =
          event.target && typeof event.target.closest === "function"
            ? event.target.closest("[data-auth-modal-close]")
            : null;
        if (closeTarget) closeAuthModal();
      });
    }
    if ($.authLoginBtn) {
      $.authLoginBtn.addEventListener("click", function () {
        void loginWithCredentials();
      });
    }
    if ($.authRegisterBtn) {
      $.authRegisterBtn.addEventListener("click", function () {
        void registerWithCredentials();
      });
    }
    if ($.authLogoutBtn) {
      $.authLogoutBtn.addEventListener("click", function () {
        void logoutAccount();
      });
    }
    if ($.authChangeAvatarBtn) {
      $.authChangeAvatarBtn.addEventListener("click", function () {
        openAuthAvatarPicker();
      });
    }
    if ($.authChangeBannerBtn) {
      $.authChangeBannerBtn.addEventListener("click", function () {
        openAuthBannerPicker();
      });
    }
    if ($.authAvatarInput) {
      $.authAvatarInput.addEventListener("change", function () {
        const file =
          $.authAvatarInput.files && $.authAvatarInput.files.length
            ? $.authAvatarInput.files[0]
            : null;
        if (!file) return;
        void uploadAccountAvatarFromFile(file);
      });
    }
    if ($.authBannerInput) {
      $.authBannerInput.addEventListener("change", function () {
        const file =
          $.authBannerInput.files && $.authBannerInput.files.length
            ? $.authBannerInput.files[0]
            : null;
        if (!file) return;
        void uploadAccountBannerFromFile(file);
      });
    }
    if ($.authPassword) {
      $.authPassword.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
          event.preventDefault();
          void loginWithCredentials();
        }
      });
    }
    if ($.authUsername) {
      $.authUsername.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
          event.preventDefault();
          if ($.authPassword) $.authPassword.focus();
        }
      });
    }

    if ($.playlistNewBtn) {
      $.playlistNewBtn.addEventListener("click", function () {
        void openPlaylistPicker(null);
      });
    }

    if ($.playlistRenameBtn) {
      $.playlistRenameBtn.addEventListener("click", function () {
        const selected = getSelectedPlaylist();
        if (!selected) {
          setPlaylistsStatus("Select a playlist first.", true);
          return;
        }
        openPlaylistRenameEditor(selected.id);
      });
    }

    if ($.playlistEditCoverBtn) {
      $.playlistEditCoverBtn.addEventListener("click", function () {
        const selected = getSelectedPlaylist();
        if (!selected) {
          setPlaylistsStatus("Select a playlist first.", true);
          return;
        }
        openPlaylistCoverPickerFor(selected.id);
      });
    }

    if ($.playlistClearBtn) {
      $.playlistClearBtn.addEventListener("click", function () {
        void clearSelectedPlaylist();
      });
    }

    if ($.playlistDeleteBtn) {
      $.playlistDeleteBtn.addEventListener("click", function () {
        void deleteSelectedPlaylist();
      });
    }

    if ($.playlistTabs) {
      $.playlistTabs.addEventListener("click", function (event) {
        const coverBtn =
          event.target && typeof event.target.closest === "function"
            ? event.target.closest("[data-playlist-cover-for]")
            : null;
        if (coverBtn) {
          event.preventDefault();
          event.stopPropagation();
          openPlaylistCoverPickerFor(coverBtn.dataset.playlistCoverFor);
          return;
        }
        const renameBtn =
          event.target && typeof event.target.closest === "function"
            ? event.target.closest("[data-playlist-rename-for]")
            : null;
        if (renameBtn) {
          event.preventDefault();
          event.stopPropagation();
          openPlaylistRenameEditor(renameBtn.dataset.playlistRenameFor);
          return;
        }
        const target =
          event.target && typeof event.target.closest === "function"
            ? event.target.closest("[data-playlist-id]")
            : null;
        if (!target) return;
        event.preventDefault();
        selectPlaylist(target.dataset.playlistId);
      });
    }

    if ($.playlistCoverInput) {
      $.playlistCoverInput.addEventListener("change", function () {
        const playlistId = String(state.playlistCoverUpload.targetPlaylistId || "").trim();
        const file =
          $.playlistCoverInput.files && $.playlistCoverInput.files.length
            ? $.playlistCoverInput.files[0]
            : null;
        if (!playlistId || !file) return;
        void uploadPlaylistCoverFromFile(playlistId, file);
      });
    }

    if ($.playlistModalClose) {
      $.playlistModalClose.addEventListener("click", function () {
        closePlaylistPicker();
      });
    }

    if ($.playlistModal) {
      $.playlistModal.addEventListener("click", function (event) {
        const target =
          event.target && typeof event.target.closest === "function"
            ? event.target.closest("[data-playlist-modal-close]")
            : null;
        if (target) closePlaylistPicker();
      });
    }

    if ($.confirmModalClose) {
      $.confirmModalClose.addEventListener("click", function () {
        closeConfirmDialog(false);
      });
    }

    if ($.confirmModalCancelBtn) {
      $.confirmModalCancelBtn.addEventListener("click", function () {
        closeConfirmDialog(false);
      });
    }

    if ($.confirmModalConfirmBtn) {
      $.confirmModalConfirmBtn.addEventListener("click", function () {
        closeConfirmDialog(true);
      });
    }

    if ($.confirmModal) {
      $.confirmModal.addEventListener("click", function (event) {
        const target =
          event.target && typeof event.target.closest === "function"
            ? event.target.closest("[data-confirm-modal-close]")
            : null;
        if (target) closeConfirmDialog(false);
      });
    }

    if ($.playlistModalCreateBtn) {
      $.playlistModalCreateBtn.addEventListener("click", function () {
        void createPlaylistFromModal();
      });
    }

    if ($.playlistModalName) {
      $.playlistModalName.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
          event.preventDefault();
          void createPlaylistFromModal();
        }
      });
    }

    if ($.playlistModalList) {
      $.playlistModalList.addEventListener("click", function (event) {
        const btn =
          event.target && typeof event.target.closest === "function"
            ? event.target.closest("[data-playlist-id]")
            : null;
        if (!btn) return;
        event.preventDefault();
        void addPendingTrackToPlaylist(btn.dataset.playlistId);
      });
    }

    if ($.searchBtn) {
      $.searchBtn.addEventListener("click", function () {
        void search();
      });
    }

    if ($.query) {
      $.query.addEventListener("keydown", function (event) {
        if (event.key === "Enter") void search();
      });
      $.query.addEventListener("input", function () {
        if (state.prefetchTimer) clearTimeout(state.prefetchTimer);
        const q = String($.query.value || "").trim();
        if (q.length < 3) return;
        state.prefetchTimer = setTimeout(function () {
          void prefetchSearch(q);
        }, 320);
      });
    }

    if ($.homeQueryForm) {
      $.homeQueryForm.addEventListener("submit", function (event) {
        event.preventDefault();
        void runHomeQueueSearch();
      });
    }
    if ($.homeQuery) {
      $.homeQuery.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          void runHomeQueueSearch();
        }
      });
    }
    $.homeChips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        if (!$.homeQuery) return;
        $.homeQuery.value = chip.dataset.homeQuery || "";
        $.homeQuery.focus();
      });
    });
    if ($.homeQueueClearBtn) {
      $.homeQueueClearBtn.addEventListener("click", function () {
        state.lists.home = [];
        state.homePaging.currentPage = 0;
        state.homeQueueSource = {
          query: "",
          endpoint: "",
          nextOffset: 0,
          exhausted: true,
          loading: false,
          pendingTracks: [],
        };
        if (state.activeListKey === "home") {
          state.queueIndex = -1;
        }
        renderHomeQueue();
        setHomeStatus("");
        setHomeQueueCaption("");
      });
    }
    if ($.homeQueuePrevBtn) {
      $.homeQueuePrevBtn.addEventListener("click", function () {
        void goToHomeQueuePage(state.homePaging.currentPage - 1);
      });
    }
    if ($.homeQueueNextBtn) {
      $.homeQueueNextBtn.addEventListener("click", function () {
        void goToHomeQueuePage(state.homePaging.currentPage + 1);
      });
    }

    $.chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        if ($.query) $.query.value = chip.dataset.query || "";
        $.query.focus();
      });
    });

    if ($.refreshFeedBtn) {
      $.refreshFeedBtn.addEventListener("click", function () {
        void loadForYou(false, { avoidCurrentBatch: true, forceDeep: true });
      });
    }

    if ($.forYouPrevBtn) {
      $.forYouPrevBtn.addEventListener("click", function () {
        void goToForYouPage(state.forYouPaging.currentPage - 1);
      });
    }
    if ($.forYouNextBtn) {
      $.forYouNextBtn.addEventListener("click", function () {
        void goToForYouPage(state.forYouPaging.currentPage + 1);
      });
    }
    window.addEventListener("resize", function () {
      const changed = updateForYouPageSize(true);
      if (changed) {
        renderForYou();
      } else {
        updateForYouStripControls();
      }
    });

    if ($.searchPrevBtn) {
      $.searchPrevBtn.addEventListener("click", function () {
        void goToSearchPage(state.searchPaging.currentPage - 1);
      });
    }
    if ($.searchNextBtn) {
      $.searchNextBtn.addEventListener("click", function () {
        void goToSearchPage(state.searchPaging.currentPage + 1);
      });
    }

    if ($.prevBtn) $.prevBtn.addEventListener("click", playPrevious);
    if ($.shuffleBtn) {
      $.shuffleBtn.addEventListener("click", function () {
        setShuffleEnabled(!state.shuffleEnabled);
      });
    }
    if ($.nextBtn) {
      $.nextBtn.addEventListener("click", function () {
        playNext(false);
      });
    }
    if ($.playBtn) $.playBtn.addEventListener("click", togglePlay);
    if ($.repeatBtn) {
      $.repeatBtn.addEventListener("click", function () {
        setRepeatOneEnabled(!state.repeatOne);
      });
    }
    if ($.likeBtn) {
      $.likeBtn.addEventListener("click", function () {
        const songId = songIdForTrack(state.current);
        if (!songId) return;
        void toggleLikeSong(songId);
      });
    }

    if ($.playerPlaylistBtn) {
      $.playerPlaylistBtn.addEventListener("click", function () {
        if (!state.current) {
          setStatus("Play a track first to add it to a playlist.", true);
          return;
        }
        void openPlaylistPicker(state.current);
      });
    }

    if ($.seek) {
      $.seek.addEventListener("pointerdown", function () {
        state.isSeeking = true;
      });

      const finishSeek = function () {
        if (!state.isSeeking) return;
        seekTo($.seek.value);
        state.isSeeking = false;
        updateProgress(true);
      };

      $.seek.addEventListener("pointerup", finishSeek);
      $.seek.addEventListener("change", finishSeek);
      $.seek.addEventListener("blur", finishSeek);
      $.seek.addEventListener("input", function () {
        updateRangeFill($.seek);
        const timing = getActivePlaybackTiming();
        const duration = Number(timing && timing.duration ? timing.duration : 0);
        if (!duration) return;
        const preview = duration * (Number($.seek.value) / 100);
        if ($.timeCurrent) $.timeCurrent.textContent = formatTime(preview);
      });
    }

    if ($.volume) {
      $.volume.addEventListener("input", function () {
        setVolume($.volume.value);
        updateRangeFill($.volume);
      });
    }

    window.addEventListener("message", function (event) {
      if (!event || event.origin !== window.location.origin) return;
      const data = event.data && typeof event.data === "object" ? event.data : null;
      if (!data || String(data.type || "") !== "spotify-auth") return;

      void (async function () {
        const status = await refreshSpotifyStatus();
        if (String(data.status || "") === "ok" && status && status.authenticated) {
          void retryPendingSpotifyFallbackAfterAuth("postmessage");
          return;
        }
        if (String(data.status || "") === "failed") {
          setStatus("Spotify login failed. YouTube fallback will continue to be used.", true);
        }
      })();
    });

    window.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && isSettingsMenuOpen()) {
        closeSettingsMenu();
      }
      if (event.key === "Escape" && state.confirmDialog.open) {
        closeConfirmDialog(false);
      }
      if (event.key === "Escape" && state.playlistPicker.open) {
        closePlaylistPicker();
      }
      if (event.key === "Escape" && $.authModal && $.authModal.classList.contains("open")) {
        closeAuthModal();
      }
    });
  }

  function bootYouTubeAPI() {
    const script = document.createElement("script");
    script.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(script);
    window.onYouTubeIframeAPIReady = initYouTubePlayer;
  }

  async function bootPersonalization() {
    await refreshAuthSession(true);
    const initialUserId =
      state.auth.authenticated && state.auth.userId
        ? state.auth.userId
        : getOrCreateUserId();
    updateIdentityUI();
    try {
      await switchIdentity(initialUserId);
    } catch (error) {}
  }

  bindEvents();
  try {
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        document.body.classList.remove("app-boot");
      });
    });
  } catch (error) {
    try {
      document.body.classList.remove("app-boot");
    } catch (innerError) {}
  }
  setImgWithFallback($.playerArt, [], PLAYER_PLACEHOLDER);
  setImgWithFallback($.rightPlayerArt, [], PLAYER_PLACEHOLDER);
  renderPlayerTrackMeta(null);
  renderHomeQueue();
  setHomeQueueCaption("");
  setShuffleEnabled(readShufflePreference());
  setRepeatOneEnabled(readRepeatOnePreference());
  updateRangeFill($.seek);
  updateRangeFill($.volume);
  switchView("home");
  bootYouTubeAPI();
  void bootPersonalization();
})();
