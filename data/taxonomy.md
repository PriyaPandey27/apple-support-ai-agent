# Apple Support Intent Taxonomy

Derived from reading real customer first-messages in `apple_conversations.json`, not invented ahead of time. Rules are simple regex keyword checks applied in a fixed priority order (first match wins) -- see `INTENT_RULES` in `prepare_dataset.py`.

## autocorrect_text_bug  (2035 examples, 8.77%)

- **Definition:** Complaints about the iOS keyboard/autocorrect bug that mangled the letter 'i' (a real, documented iOS 11.1-era defect).
- **Include if:** Message mentions the corrupted 'i' character, 'autocorrect', or describes the keyboard glitch.
- **Exclude if:** General typing complaints unrelated to the specific 'i' bug.

## account_security_icloud  (927 examples, 4.0%)

- **Definition:** Login, Apple ID, iCloud, password, or account-security/lockout issues.
- **Include if:** Mentions iCloud, Apple ID, password, login, 2FA, or account being locked/hacked.
- **Exclude if:** Purchase/billing issues tied to an account (goes to billing_repair_order).

## billing_repair_order  (578 examples, 2.49%)

- **Definition:** Refunds, incorrect charges, orders, warranty, repairs, or replacements.
- **Include if:** Mentions refund, charge, order, warranty, repair, replacement, return, Genius Bar.
- **Exclude if:** General battery or screen complaints with no billing/repair request.

## battery_power  (2350 examples, 10.13%)

- **Definition:** Battery drain, charging problems, or device not powering on.
- **Include if:** Mentions battery, charging, draining, or the device failing to turn on.
- **Exclude if:** Device turning off/crashing without a battery/charging cause (device_not_working).

## screen_hardware_damage  (1303 examples, 5.62%)

- **Definition:** Cracked, broken, or malfunctioning screen/display or physical buttons.
- **Include if:** Mentions screen, crack, display, Touch ID/Face ID, stuck buttons, water damage.
- **Exclude if:** Software-only display glitches with no hardware damage implied.

## connectivity  (946 examples, 4.08%)

- **Definition:** Wi-Fi, Bluetooth, cellular signal, or hotspot connection problems.
- **Include if:** Mentions wifi, bluetooth, signal, cellular, hotspot, or 'won't connect'.
- **Exclude if:** App-specific connectivity (e.g. one app not loading) with no network complaint.

## ios_software_update  (5763 examples, 24.85%)

- **Definition:** Problems following or caused by an iOS/software update.
- **Include if:** Mentions update, iOS version numbers, software, reinstalling.
- **Exclude if:** The specific autocorrect 'i' bug (its own category).

## app_appstore_media  (1489 examples, 6.42%)

- **Definition:** Issues with the App Store, iTunes, Apple Music, or a specific app/download.
- **Include if:** Mentions App Store, iTunes, Apple Music, an app, or downloads/podcasts.
- **Exclude if:** OS-level update issues with no specific app mentioned.

## device_not_working  (415 examples, 1.79%)

- **Definition:** Device frozen, crashing, restarting, or generally unresponsive, without a clear battery/screen/connectivity cause.
- **Include if:** Mentions frozen, crashing, not responding, bricked, dead.
- **Exclude if:** Cases already covered by a more specific category above.

## other_unclear  (7387 examples, 31.85%)

- **Definition:** Everything that doesn't match a rule above: vague complaints, off-topic mentions, single-word replies, or messages needing more context.
- **Include if:** No keyword match from any rule above.
- **Exclude if:** n/a (fallback bucket).

## Sample messages per intent

**autocorrect_text_bug**
- "OKAY now I need @115858 to get their shit together and fix this “i” problem because I’m sick of not understanding peoples shit when they type the letter I"
- "@AppleSupport why does this appear every time I️ write the I️? https://t.co/EN5xZKTTT0"
- "@AppleSupport every time “eye” type “eye” it’s turns to I️... HELP!"

**account_security_icloud**
- "@AppleSupport having a very persistent issue where after I delete a folder from the Notes app, it continues to reappear on all synced iCloud devices"
- "Hey @116333 while you’re selling £1149 phones you have websites where you can’t login properly on iPhones. https://t.co/gZdlGVBNy8"
- "@115858 is there anyway you can send the verification code to my apple ID to activate my phone to my email? My phone is completely broke!"

**billing_repair_order**
- "bravo @115858 - after completing my order with store pickup, tells me no stores available but won't let me ship it. Try again, get this: https://t.co/YglVYhTkxP"
- "Why did the #iPhoneUpdate completely change the #podcastapp ? It’s terrible now 😑😡 I shouldn’t have to open the app every time a podcast ends to start a new one"
- "@115858 you’re pants. Sent from my rubbish iPhone because the ‘new’ iPad you gave to me today to replace the broken 1 year and 5 days old one, doesn’t work eith"

**battery_power**
- "@115858 repeats itself.! Again some shitty update n phones battery gone for a toss #ios #ios11 #ios1111"
- "@116333 @34202 Since ios 11 update not such solution on my phone springboard i send pic and mssg too suck my battery n phone also gettin warm solution pls"
- "@173329 @115858 Serious battery life issues in IOS 11"

**screen_hardware_damage**
- "In love with my beautiful #iPhoneX 😍 just arrived today. It'd be perfect if it weren't for that grueling speaker crackling and rattling. @115858 @116333 pls tel"
- "@AppleSupport what's the best iPhone X case? Waterproof, dropproof &amp; has a screen protector (possibly one that's already built-in the case..."
- "@AppleSupport nightmare update 11 !! AirDrop moved and no 4G brand new iPhone 5se now cracks up on calls !! Horrendous !! Pls sort ASAP"

**connectivity**
- "QC35 bluetooth fail to connect in latest version iOS 11.2. Any updates?  @115858 @4770 ?"
- "Hey @AppleSupport can you please explain why ios11 no longer let's my phone connect to my car via Bluetooth????"
- "@AppleSupport i have tried restarting, still nothing. also occurs on wifi"

**ios_software_update**
- "@115858 my series 1 watch stopped working with the new updates. Won’t pair with my phone. None of the stores in my area have appt. less than 2hr wait time. What"
- "Downloaded iOS 11 and now my iPod touch won't play any music. Brilliant stuff, @AppleSupport."
- "@AppleSupport hi, since updating to ios11 I now can’t delete any photos off my iPhone! Is this a glitch in the update? During the update it synced somehow and I"

**app_appstore_media**
- "@AppleSupport how can I stop apps appearing like this after I've used them on my iPad? #apple #ipad https://t.co/FDNloYoZtO"
- "@AppleSupport my notifications are not appearing, I'm going crazy to have to open and close apps at all times"
- "@AppleSupport are you doing 12 days of Christmas this year the one where you get a different app/song for free ?"

**device_not_working**
- "My iMovie constantly keeps crashing. Won’t even start! :O Whaaaat @AppleSupport"
- "My iPhone keeps shutting off and deleting stuff. ETF @AppleSupport 😡😡 I'm keeping my 6s plus! Or maybe I'll go back to using an Android 🤔"
- "@115858 @116333 @119462 My reservation doesn’t work for #iPhoneX. I’ve called Support 6 times and they don’t know what to do! HELP 🆘😭 https://t.co/vfoXf5Kccl"

**other_unclear**
- "Hey @AppleSupport bug reporter isn’t working…"
- "@115858  my ipad needs help!"
- "@115858 OH COME ON https://t.co/rho54WNdQK"

